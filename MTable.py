import numpy as np
import math
from collections.abc import Callable
import warnings
from typing import List, Optional, Tuple, Union
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import inspect

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
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from transformers.integrations.fsdp import is_fsdp_managed_module

from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint

from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)


from transformers.generation.configuration_utils import (
    NEED_SETUP_CACHE_CLASSES_MAPPING,
    QUANT_BACKEND_CLASSES_MAPPING,
    GenerationConfig,
    GenerationMode,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)


from transformers.generation.logits_process import LogitsProcessorList

import torch.distributed as dist


from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    logging,
)



# 创建一个logger对象
logger = logging.get_logger(__name__)
# 设置日志级别
logger.setLevel(logging.WARNING)




class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        # --- Table Injection Parameters ---
        self.apply_table_injection = False
        # Store the sequence embedding directly, not adapter weights
        self.table_token = None  
        self.table_embedding = None # Shape: (SeqLen_table, Hidden)
        self.retracing_ratio = 0.05 # Example: Blending factor for the attention output
        self.entropy_threshold = 1.0
        self.starting_layer = 0
        self.ending_layer = 32
        self.adapt_signal = 0


        # Hyperparameter for non-learnable attention
        self.attention_temperature = self.hidden_size**0.5

    def initialize_table_embedding(self, table_embedding):
        """Stores the processed table embedding sequence. Non-learnable."""

        ref_param = self.gate_proj.weight
        target_device = ref_param.device
        target_dtype = ref_param.dtype

        table_embedding = table_embedding.to(target_device, dtype=target_dtype)

        if table_embedding.ndim == 3 and table_embedding.shape[0] == 1:
            # Squeeze batch dim: Result shape (SeqLen_table, H)
            self.table_seq_embedding = table_embedding.squeeze(0)
            # print(f"DEBUG: Stored table_seq_embedding with shape {self.table_seq_embedding.shape}")

    def forward(self, x): # x shape: (Batch, SeqLen_x, Hidden)

        if hasattr(self.config, 'pretraining_tp') and self.config.pretraining_tp > 1:
            print("Warning: Tensor Parallel > 1 detected. Injection logic might need adjustment. Falling back.")
            ffn_intermediate = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            down_proj = self.down_proj(ffn_intermediate)
            return down_proj 


        # --- Injection Logic ---
        # Check if injection is active and table embedding is available
        if self.adapt_signal == 1 and self.table_seq_embedding is not None:
            # --- Non-Learnable Cross-Attention Path ---
            try:
                # 1. Calculate standard FFN output (can potentially be optimized later)
                ffn_intermediate = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                ffn_out = self.down_proj(ffn_intermediate)

                # 2. Prepare query (x) and key/value (table_seq_embedding)
                query = x # (B, S_x, H)
                key = self.table_seq_embedding # (S_table, H)
                value = self.table_seq_embedding # (S_table, H)

                # Ensure key/value are on the same device as query
                key = key.to(query.device, dtype=query.dtype)
                value = value.to(query.device, dtype=query.dtype)

                # 3. Calculate Similarity Scores (using Cosine Similarity)
                # Normalize vectors for cosine similarity calculation
                query_norm = F.normalize(query, p=2, dim=-1) # (B, S_x, H)
                key_norm = F.normalize(key, p=2, dim=-1)   # (S_table, H)

                # Calculate batched dot product for similarity: (B, S_x, H) @ (H, S_table) -> (B, S_x, S_table)
                # Transpose key_norm for matmul
                similarity_scores = torch.matmul(query_norm, key_norm.t()) # Result shape (B, S_x, S_table)

                # 4. Get Attention Weights via Softmax
                # Scale scores by temperature before softmax for better distribution
                attention_weights = F.softmax(similarity_scores / self.attention_temperature, dim=-1)
                # attention_weights shape: (B, S_x, S_table)

                # 5. Compute Weighted Sum of Values
                # Need value shape (S_table, H). Expand for batch matmul if needed: (B, S_table, H)
                value_expanded = value.unsqueeze(0).expand(x.shape[0], -1, -1) # (B, S_table, H)
                # Weighted sum: (B, S_x, S_table) @ (B, S_table, H) -> (B, S_x, H)
                attn_output = torch.matmul(attention_weights, value_expanded)

                # 6. Combine FFN output and Attention output
                # Simple blending using the retracing ratio
                # Ensure attn_output is scaled appropriately if needed, but direct blending is common
                down_proj = (ffn_out * (1 - self.retracing_ratio) + attn_output * self.retracing_ratio)

                # Optional: Add Layer Normalization after blending (using a pre-defined LN if desired)
                # if hasattr(self, 'output_layer_norm'): # Check if you added one in __init__
                #    down_proj = self.output_layer_norm(down_proj)

            except Exception as e:
                print(f"Error during non-learnable cross-attention: {e}. Falling back to normal FFN.")
                # Fallback to normal FFN path on error
                ffn_intermediate = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                down_proj = self.down_proj(ffn_intermediate)

        else:
            # (If adapt_signal is 0 or table_seq_embedding is None)
            ffn_intermediate = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            down_proj = self.down_proj(ffn_intermediate)

        return down_proj

class GlmMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(self.hidden_size, 2*self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        # --- Table Injection Parameters ---
        self.apply_table_injection = False
        # Store the sequence embedding directly, not adapter weights
        self.table_token = None  
        self.table_embedding = None # Shape: (SeqLen_table, Hidden)
        self.retracing_ratio = 0.05 # Example: Blending factor for the attention output
        self.entropy_threshold = 1.0
        self.starting_layer = 0
        self.ending_layer = 32
        self.adapt_signal = 0


        # Hyperparameter for non-learnable attention
        self.attention_temperature = self.hidden_size**0.5

    def initialize_table_embedding(self, table_embedding):
        """Stores the processed table embedding sequence. Non-learnable."""

        ref_param = self.gate_up_proj.weight
        target_device = ref_param.device
        target_dtype = ref_param.dtype

        table_embedding = table_embedding.to(target_device, dtype=target_dtype)

        if table_embedding.ndim == 3 and table_embedding.shape[0] == 1:
            # Squeeze batch dim: Result shape (SeqLen_table, H)
            self.table_seq_embedding = table_embedding.squeeze(0)
            # print(f"DEBUG: Stored table_seq_embedding with shape {self.table_seq_embedding.shape}")

    def forward(self, x): # x shape: (Batch, SeqLen_x, Hidden)



        if self.adapt_signal == 1 and self.table_seq_embedding is not None:
            # --- Non-Learnable Cross-Attention Path ---
            try:
                up_states = self.gate_up_proj(x)
                
                gate, up_states = up_states.chunk(2,dim=-1)
                up_states = up_states  * self.act_fn(gate)

                ffn_out = self.down_proj(up_states)

                query = x # (B, S_x, H)
                key = self.table_seq_embedding # (S_table, H)
                value = self.table_seq_embedding # (S_table, H)

                key = key.to(query.device, dtype=query.dtype)
                value = value.to(query.device, dtype=query.dtype)

                query_norm = F.normalize(query, p=2, dim=-1) # (B, S_x, H)
                key_norm = F.normalize(key, p=2, dim=-1)   # (S_table, H)

                similarity_scores = torch.matmul(query_norm, key_norm.t()) # Result shape (B, S_x, S_table)

                attention_weights = F.softmax(similarity_scores / self.attention_temperature, dim=-1)

                value_expanded = value.unsqueeze(0).expand(x.shape[0], -1, -1) # (B, S_table, H)
                attn_output = torch.matmul(attention_weights, value_expanded)

 
                down_proj = (ffn_out * (1 - self.retracing_ratio) + attn_output * self.retracing_ratio)

 

            except Exception as e:
                print(f"Error during non-learnable cross-attention: {e}. Falling back to normal FFN.")
                # Fallback to normal FFN path on error
                ffn_intermediate = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                down_proj = self.down_proj(ffn_intermediate)

        else:
            up_states = self.gate_up_proj(x)
                
            gate, up_states = up_states.chunk(2,dim=-1)
            up_states = up_states  * self.act_fn(gate)

            down_proj = self.down_proj(up_states)

        return down_proj


class MistralMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        # --- Table Injection Parameters ---
        self.apply_table_injection = False
        self.table_token = None
        self.table_embedding = None  # Shape: (SeqLen_table, Hidden)
        self.retracing_ratio = 0.05  # Blending factor for the attention output
        self.entropy_threshold = 1.0
        self.starting_layer = 0
        self.ending_layer = 32
        self.adapt_signal = 0

        self.attention_temperature = self.hidden_size**0.5

    def initialize_table_embedding(self, table_embedding):
        """Stores the processed table embedding sequence. Non-learnable."""
        ref_param = self.gate_proj.weight
        target_device = ref_param.device
        target_dtype = ref_param.dtype

        table_embedding = table_embedding.to(target_device, dtype=target_dtype)

        if table_embedding.ndim == 3 and table_embedding.shape[0] == 1:
            # Squeeze batch dim: Result shape (SeqLen_table, H)
            self.table_seq_embedding = table_embedding.squeeze(0)

    def forward(self, x):  # x shape: (Batch, SeqLen_x, Hidden)
        if hasattr(self.config, 'pretraining_tp') and self.config.pretraining_tp > 1:
            print("Warning: Tensor Parallel > 1 detected. Injection logic might need adjustment. Falling back.")
            ffn_intermediate = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            down_proj = self.down_proj(ffn_intermediate)
            return down_proj

        # --- Injection Logic ---
        if self.adapt_signal == 1 and hasattr(self, 'table_seq_embedding') and self.table_seq_embedding is not None:
            try:
                ffn_intermediate = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                ffn_out = self.down_proj(ffn_intermediate)

                query = x  # (B, S_x, H)
                key = self.table_seq_embedding  # (S_table, H)
                value = self.table_seq_embedding  # (S_table, H)

                # Ensure key/value are on the same device as query
                key = key.to(query.device, dtype=query.dtype)
                value = value.to(query.device, dtype=query.dtype)

                query_norm = F.normalize(query, p=2, dim=-1)  # (B, S_x, H)
                key_norm = F.normalize(key, p=2, dim=-1)    # (S_table, H)
                similarity_scores = torch.matmul(query_norm, key_norm.t())

                attention_weights = F.softmax(similarity_scores / self.attention_temperature, dim=-1)

                value_expanded = value.unsqueeze(0).expand(x.shape[0], -1, -1)  # (B, S_table, H)
                attn_output = torch.matmul(attention_weights, value_expanded)

                down_proj = (ffn_out * (1 - self.retracing_ratio) + attn_output * self.retracing_ratio)

            except Exception as e:
                print(f"Error during non-learnable cross-attention: {e}. Falling back to normal FFN.")
                ffn_intermediate = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                down_proj = self.down_proj(ffn_intermediate)
        else:
            ffn_intermediate = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            down_proj = self.down_proj(ffn_intermediate)

        return down_proj


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
        **flash_attn_kwargs,
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
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )


    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # --- Attention Mask --- (Keep standard logic)
    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
    # embed positions
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # --- Decoder Layers ---
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    # --- Table Injection Setup ---
    # Get global settings from the first layer's MLP
    first_mlp = self.layers[0].mlp
    apply_table_injection = self.layers[0].mlp.apply_table_injection  # Check if injection is enabled
    
    # 检查是否有表格内容，如果没有则禁用表格注入
    has_table_content = hasattr(first_mlp, 'table_token') and first_mlp.table_token is not None
    if not has_table_content:
        apply_table_injection = False

    entropy_threshold = getattr(first_mlp, 'entropy_threshold', 1.0)
    starting_layer = getattr(first_mlp, 'starting_layer', 0)
    ending_layer = getattr(first_mlp, 'ending_layer', len(self.layers))
    retracing_ratio = getattr(first_mlp, 'retracing_ratio', 0.0) # Need this for initialization scale
    table_retracing_event = False # Flag to ensure trigger happens only once per forward pass

    for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
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
                    position_embeddings,
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
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)
            
        # --- Entropy Calculation and Trigger Logic ---
        # 只有当apply_table_injection为True且有表格内容时才进行熵计算和触发逻辑
        if (apply_table_injection and has_table_content and 
            not table_retracing_event and 
            layer_idx >= starting_layer and 
            layer_idx <= ending_layer): 
            
            norm_hidden_states = self.norm(hidden_states)
            logits = self.lm_head(norm_hidden_states)
            last_token_logits = logits[:, -1, :]
            last_token_logits = last_token_logits.float()
            last_token_logits = self.logits_processor(input_ids, last_token_logits)

            k_topk = 5
            top_k_scores, _ = torch.topk(last_token_logits, k_topk)
            epsilon = 1e-9

            # 应用温度参数使分布更加平滑或尖锐
            temperature = 1.0  #  >1 更平滑，<1 更尖锐
            scaled_scores = top_k_scores / temperature

            probabilities = F.softmax(scaled_scores, dim=-1)

            entropy = -torch.sum(probabilities * torch.log2(probabilities + epsilon))
            normalized_entropy = entropy / math.log2(k_topk)
            avg_entropy = normalized_entropy.item()


            if avg_entropy > entropy_threshold:
                table_retracing_event = True 
                next_layer_mlp = self.layers[layer_idx + 1].mlp

                current_table_token = self.layers[0].mlp.table_token.to(input_ids.device)  # Retrieve shared token
                table_embeds = self.embed_tokens(current_table_token)
                next_layer_mlp.initialize_table_embedding(table_embeds)
                # print(f"add table feature to layer {layer_idx},avg_entropy: {avg_entropy}")
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

    output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
    )
    return output if return_dict else output.to_tuple()



@torch.no_grad()
def generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    use_model_defaults: Optional[bool] = None,
    *,  # Force subsequent args to be keyword args
    table_token: Optional[torch.Tensor] = None,  # New argument for table content
    **kwargs,
    ):

    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    self._validate_model_class()
    tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
    assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)  # only used for assisted generation

    if "table_token" in kwargs:
        if table_token is None:  # Prioritize direct argument
            table_token = kwargs.pop("table_token")
        else:
            kwargs.pop("table_token")  # Remove if passed redundantly

    # 处理表格内容，如果为None，则清除之前的表格内容
    if table_token is not None:
        logger.info("Processing table_token for injection...")
        self.model.layers[0].mlp.table_token = table_token
    else:
        # 如果表格内容为None，清除之前的表格内容
        logger.info("No table_token provided, disabling table injection...")
        if hasattr(self.model.layers[0].mlp, 'table_token'):
            self.model.layers[0].mlp.table_token = None
            self.model.layers[0].mlp.table_seq_embedding = False
            

    generation_config, model_kwargs = self._prepare_generation_config(
        generation_config, use_model_defaults, **kwargs
    )
    self._validate_model_kwargs(model_kwargs.copy())
    self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

    # 2. Set generation parameters if not already defined
    if synced_gpus is None:
        synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1

    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    self.model.logits_processor = logits_processor
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs
    kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

    # 3. Define model inputs
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    device = inputs_tensor.device
    self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

    # decoder-only models must use left-padding for batched generation.
    if not self.config.is_encoder_decoder:
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        if (
            generation_config._pad_token_tensor is not None
            and batch_size > 1
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    # 4. Define other model kwargs
    if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        generation_config.use_cache = True

    if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config, model_kwargs
        )
    elif kwargs_has_attention_mask:
        if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
            raise ValueError("`attention_mask` passed to `generate` must be 2D.")

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # If model is encoder-decoder, encoder_outputs are created and added to `model_kwargs`
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    if generation_config.token_healing:
        input_ids = self.heal_tokens(input_ids, tokenizer)

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
    generation_config = self._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        inputs_tensor=inputs_tensor,
        input_ids_length=input_ids_length,
    )

    # 7. Prepare the cache.
    max_cache_length = generation_config.max_length - 1
    if (
        inputs_tensor.shape[1] != input_ids_length
        and model_input_name == "inputs_embeds"
        and not self.config.is_encoder_decoder
    ):
        max_cache_length += inputs_tensor.shape[1]
    self._prepare_cache_for_generation(
        generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
    )

    # 8. Determine generation mode
    generation_mode = generation_config.get_generation_mode(assistant_model)

    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError(
            "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        )

    if self.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 9. Prepare logits processors and stopping criteria
    prepared_logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        device=inputs_tensor.device,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )
    prepared_stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
    )

    # 10. go into different generation modes
    if generation_mode == GenerationMode.ASSISTED_GENERATION:
        if generation_config.num_return_sequences > 1:
            raise ValueError(
                "num_return_sequences has to be 1 when doing assisted generate, "
                f"but is {generation_config.num_return_sequences}."
            )
        if batch_size > 1:
            raise ValueError("assisted generate is only supported for batch_size = 1")
        if not model_kwargs["use_cache"]:
            raise ValueError("assisted generate requires `use_cache=True`")
        if generation_config.cache_implementation in ["static", "hybrid", "sliding_window"]:
            raise ValueError("assisted generate is not supported with Static cache classes`")
        if self._is_stateful:
            # In assisted generation we need the ability to confirm whether the model would pick certain tokens,
            # which is not possible with stateful models (they can't reset to a previous subset of generated text)
            raise ValueError(
                f"assisted generation is not supported with stateful models, such as {self.__class__.__name__}"
            )

        # 11. Get the candidate generator, given the parameterization
        candidate_generator = self._get_candidate_generator(
            generation_config=generation_config,
            input_ids=input_ids,
            inputs_tensor=inputs_tensor,
            assistant_model=assistant_model,
            logits_processor=logits_processor,
            target_tokenizer=tokenizer,
            assistant_tokenizer=assistant_tokenizer,
            model_kwargs=model_kwargs,
        )

        # 12. run assisted generate
        result = self._assisted_decoding(
            input_ids,
            candidate_generator=candidate_generator,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )
    elif generation_mode == GenerationMode.DOLA_GENERATION:
        if self._is_stateful:
            # DoLa decoding was not designed for stateful models, and would require some changes
            raise ValueError(
                f"dola decoding is not supported with stateful models, such as {self.__class__.__name__}"
            )
        result = self._dola_decoding(
            input_ids,
            dola_layers=generation_config.dola_layers,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
        if not model_kwargs["use_cache"]:
            raise ValueError("Contrastive search requires `use_cache=True`")
        if self._is_stateful:
            # Just like assisted generation, we need to be able to rollback to a previous state (see comment above)
            raise ValueError(
                f"contrastive search is not supported with stateful models, such as {self.__class__.__name__}"
            )

        result = self._contrastive_search(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
        # 11. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
        result = self._sample(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
        # 11. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        # 12. run beam sample
        result = self._beam_search(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
        # 11. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            num_beam_groups=generation_config.num_beam_groups,
            max_length=generation_config.max_length,
        )
        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        # 13. run beam search
        result = self._group_beam_search(
            input_ids,
            beam_scorer,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
        final_constraints = []
        if generation_config.constraints is not None:
            final_constraints = generation_config.constraints

        if generation_config.force_words_ids is not None:

            def typeerror():
                raise ValueError(
                    "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
                    f"of positive integers, but is {generation_config.force_words_ids}."
                )

            if (
                not isinstance(generation_config.force_words_ids, list)
                or len(generation_config.force_words_ids) == 0
            ):
                typeerror()

            for word_ids in generation_config.force_words_ids:
                if isinstance(word_ids[0], list):
                    if not isinstance(word_ids, list) or len(word_ids) == 0:
                        typeerror()
                    if any(not isinstance(token_ids, list) for token_ids in word_ids):
                        typeerror()
                    if any(
                        any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                        for token_ids in word_ids
                    ):
                        typeerror()

                    constraint = DisjunctiveConstraint(word_ids)
                else:
                    if not isinstance(word_ids, list) or len(word_ids) == 0:
                        typeerror()
                    if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                        typeerror()

                    constraint = PhrasalConstraint(word_ids)
                final_constraints.append(constraint)

        # 11. prepare beam search scorer
        constrained_beam_scorer = ConstrainedBeamSearchScorer(
            constraints=final_constraints,
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )
        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        # 13. run beam search
        result = self._constrained_beam_search(
            input_ids,
            constrained_beam_scorer=constrained_beam_scorer,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    # Convert to legacy cache format if requested
    if (
        generation_config.return_legacy_cache is True
        and hasattr(result, "past_key_values")
        and getattr(result.past_key_values, "to_legacy_cache") is not None
    ):
        result.past_key_values = result.past_key_values.to_legacy_cache()

    return result

def apply_table_function_llama():
    transformers.models.llama.modeling_llama.LlamaMLP = LlamaMLP
    transformers.models.llama.modeling_llama.LlamaModel.forward = table_forward
    transformers.models.llama.modeling_llama.LlamaForCausalLM.generate = generate

def apply_table_function_qwen():
    transformers.models.qwen2.modeling_qwen2.Qwen2MLP = LlamaMLP
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = table_forward
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.generate = generate

def apply_table_function_glm():
    transformers.models.glm.modeling_glm.GlmMLP = GlmMLP
    transformers.models.glm.modeling_glm.GlmModel.forward = table_forward
    transformers.models.glm.modeling_glm.GlmForCausalLM.generate = generate

def apply_table_function_mistral():
    transformers.models.mistral.modeling_mistral.MistralMLP = MistralMLP
    transformers.models.mistral.modeling_mistral.MistralModel.forward = table_forward
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.generate = generate


def apply_table_llama(
        self,
        starting_layer: int,
        ending_layer: int,
        entropy_threshold: float,
        retracing_ratio: float
    ):
    self.model.lm_head = self.lm_head
    self.model.layers[0].mlp.apply_table_injection = True
    self.model.layers[0].mlp.starting_layer = starting_layer
    self.model.layers[0].mlp.ending_layer = ending_layer
    self.model.layers[0].mlp.entropy_threshold = entropy_threshold
    for layer in range(len(self.model.layers)):
        self.model.layers[layer].mlp.retracing_ratio = retracing_ratio

