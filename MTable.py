import numpy as np
import math
import warnings
from typing import List, Optional, Tuple, Union
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import transformers

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.generation.logits_process import LogitsProcessorList



class LlamaMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        #输入的隐藏状态映射到中间维度intermediate_size  -- 作用是生成一个门控信号
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        #对输入进行升维
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        #对输入进行降维到原始的隐藏状态
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        # --- Table Injection Parameters ---
        self.apply_table_injection = False # Global flag (set during setup)
        self.table_token = None         # Will hold the precomputed table embedding (set per-input)
        self.retracing_ratio = 0        # Injection strength (set during setup)
        self.entropy_threshold = 1.0    # Trigger threshold (set during setup)
        self.starting_layer = 0         # Layer range start (set during setup)
        self.ending_layer = 32          # Layer range end (set during setup, adjust based on model)
        self.adapt_signal = 0           # 0: Normal FFN, 1: Inject Table Info

        # Adapter weights - initialized when triggered
        self.adpt_w1 = None
        self.adpt_w2 = None
        # --- End Table Injection Parameters ---
    
    def forward(self,x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        elif self.adapt_signal == 0:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        
        elif self.adapt_signal == 1:
            # --- Table Injection Path ---
            # Ensure adapter weights are ready (should have been set by trigger logic)
            if self.adpt_w1 is None or self.adpt_w2 is None:
                 print(f"Warning: adapt_signal==1 but adapter weights not initialized in MLP. Falling back to normal FFN.")
                 down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            else:
                # 1. Calculate standard FFN output
                ffn_out = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

                # 2. Calculate adapter output using adapter weights
                self.adpt_w1 = self.adpt_w1.to(x.device)
                self.adpt_w2 = self.adpt_w2.to(x.device)
                adapter_out = torch.matmul(torch.matmul(x, self.adpt_w1.T), self.adpt_w2.T) # (B, S, H) -> (B, S, I) -> (B, S, H)

                # 3. Normalize adapter output magnitude
                epsilon = 1e-6
                ffn_mean_abs = torch.mean(torch.abs(ffn_out)).clamp(min=epsilon)
                adapter_mean_abs = torch.mean(torch.abs(adapter_out)).clamp(min=epsilon)
                norm_adapter_out = (ffn_mean_abs / adapter_mean_abs) * adapter_out
                # 4. Blend FFN output and normalized adapter output
                down_proj = (ffn_out * (1 - self.retracing_ratio) + norm_adapter_out * self.retracing_ratio)

        return down_proj
    def initialize_adapter_weights(self, table_embedding):
        """Initializes adapter weights based on the table embedding."""
        if table_embedding is None:
            self.adpt_w1 = None
            self.adpt_w2 = None
            return

        table_embedding = table_embedding.to(self.gate_proj.weight.device)

        scale_factor1 = (torch.mean(torch.abs(self.up_proj.weight))) / (torch.mean(torch.abs(table_embedding)).clamp(min=1e-6))
        scale_factor2 = (torch.mean(torch.abs(self.down_proj.weight))) / (torch.mean(torch.abs(table_embedding)).clamp(min=1e-6))

 
        init_w1 = (scale_factor1 * table_embedding).unsqueeze(0).repeat(self.hidden_size, 1)
        init_w2 = (scale_factor2 * table_embedding).unsqueeze(1).repeat(1, self.hidden_size)

        self.adpt_w1 = init_w1 # Shape (H, H) -> Matmul needs (H, H).T
        self.adpt_w2 = init_w2 # Shape (H, H) -> Matmul needs (H, H).T


def table_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # --- Table Injection Argument ---
        table_embedding: Optional[torch.Tensor] = None, # Pass the precomputed table embedding here
        lm_head: Optional[nn.Module] = None # Pass the LM head for entropy calculation
    ) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
    # --- Retrieve Embedding ---
    if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
   
    # --- Handle Cache and Position IDs --- (Keep standard logic)
    past_seen_tokens = 0
    if use_cache:  # kept for BC (cache positions)
        if not isinstance(past_key_values, StaticCache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

    if cache_position is None:
        if isinstance(past_key_values, StaticCache):
            raise ValueError("cache_position is a required argument when using StaticCache.")
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # --- Attention Mask --- (Keep standard logic)
    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)

    # embed positions
    hidden_states = inputs_embeds

    # --- Decoder Layers ---
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    # --- Table Injection Setup ---
    # Get global settings from the first layer's MLP
    first_mlp = self.layers[0].mlp
    apply_injection = getattr(first_mlp, "apply_table_injection", False)  # Check if injection is enabled

    if apply_injection and table_embedding is None:
        pass
    if apply_injection and table_embedding is not None:
        self.layers[0].mlp.table_token = table_embedding.to(hidden_states.device)
    
     # Get other shared parameters
    entropy_threshold = getattr(first_mlp, 'entropy_threshold', 1.0)
    starting_layer = getattr(first_mlp, 'starting_layer', 0)
    ending_layer = getattr(first_mlp, 'ending_layer', len(self.layers))
    retracing_ratio = getattr(first_mlp, 'retracing_ratio', 0.0) # Need this for initialization scale

    table_retracing_event = False # Flag to ensure trigger happens only once per forward pass


    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
            )
        else:
            layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        # --- Entropy Calculation and Trigger Logic ---
        if apply_injection and not table_retracing_event and layer_idx >= starting_layer and layer_idx <=ending_layer : # Check if within range AND not the very last layer
            norm_hidden_states = self.norm(hidden_states)
            logits = lm_head(norm_hidden_states)
            last_token_logits = logits[:, -1, :]
            last_token_logits = last_token_logits.float() # Ensure float32 for stable softmax/log

            # Calculate entropy (using top-k approximation like LLaVA for efficiency)
            k_topk = 10  
            top_k_scores, _ = torch.topk(last_token_logits, k_topk)
            probabilities  = F.softmax(top_k_scores, dim=-1)

            epsilon_log = 1e-9
            entropy = -torch.sum(probabilities * torch.log(probabilities.clamp(min=epsilon_log)), dim=-1)
            entropy = entropy / np.log(k_topk)
            avg_entropy = torch.mean(entropy).item()
            
            if avg_entropy > entropy_threshold:
                table_retracing_event = True 
                next_layer_mlp = self.layers[layer_idx + 1].mlp

                current_table_token = self.layers[0].mlp.table_token # Retrieve shared token

                next_layer_mlp.initialize_adapter_weights(current_table_token)
                next_layer_mlp.adapt_signal = 1
                next_layer_mlp.retracing_ratio = retracing_ratio
        
        if decoder_layer.mlp.adapt_signal == 1:
            decoder_layer.mlp.adapt_signal = 0
            decoder_layer.mlp.adpt_w1 = None
            decoder_layer.mlp.adpt_w2 = None



    hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
        )
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
    )



def apply_table_llama(
        self,
        starting_layer: int,
        ending_layer: int,
        entropy_threshold: float,
        retracing_ratio: float
    ):
    transformers.models.llama.modeling_llama.LlamaMLP = LlamaMLP
    transformers.models.llama.modeling_llama.LlamaModel.forward = forward

    self.model.lm_head = self.lm_head
    self.model.layers[0].mlp.apply_memvr = True
    self.model.layers[0].mlp.starting_layer = starting_layer
    self.model.layers[0].mlp.ending_layer = ending_layer
    self.model.layers[0].mlp.entropy_threshold = entropy_threshold
    for layer in range(31):
        self.model.layers[layer].mlp.retracing_ratio = retracing_ratio
