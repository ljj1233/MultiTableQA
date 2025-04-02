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

    def initialize_adapter_weights(self,table_embedding):
        """Initializes adapter weights based on the table embedding."""
        if table_embedding is None:
            self.adpt_w1 = None
            self.adpt_w2 = None
            return

        table_embedding = self.table_embedding.to(self.gate_proj.weight.device)

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
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
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

    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    # --- Decoder Layers ---
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    # --- Table Injection Setup ---
    # Get global settings from the first layer's MLP

    # table_embedding = kwargs.get("table_embedding", None)
    # if apply_injection and table_embedding is not None:
    #     self.layers[0].mlp.table_token = table_embedding.to(hidden_states.device)

    first_mlp = self.layers[0].mlp
    apply_injection = getattr(first_mlp, "apply_table_injection", False)  # Check if injection is enabled
    entropy_threshold = getattr(first_mlp, 'entropy_threshold', 1.0)
    starting_layer = getattr(first_mlp, 'starting_layer', 0)
    ending_layer = getattr(first_mlp, 'ending_layer', len(self.layers))
    retracing_ratio = getattr(first_mlp, 'retracing_ratio', 0.0) # Need this for initialization scale
    table_token = getattr(first_mlp, 'table_token', None)
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

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        # --- Entropy Calculation and Trigger Logic ---
        if apply_injection and not table_retracing_event and layer_idx >= starting_layer and layer_idx <=ending_layer : # Check if within range AND not the very last layer
            norm_hidden_states = self.norm(hidden_states)
            logits = self.lm_head(norm_hidden_states)
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
                table_embedding = self.embed_tokens(table_token) 
                # Set the table embedding for the next layer's MLP
                next_layer_mlp.initialize_adapter_weights(table_embedding)
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
        *, # Force subsequent args to be keyword args
        table_content: Optional[str] = None, # <--- OUR NEW ARGUMENT
        **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:

    # 1. Handle `generation_config` and kwargs, validate call
    self._validate_model_class()

    if "table_content" in kwargs:
        if table_content is None: # Prioritize direct argument
            table_content = kwargs.pop("table_content")
        else:
            kwargs.pop("table_content") # Remove if passed redundantly


    generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
    self._validate_model_kwargs(model_kwargs.copy())

    if synced_gpus is None:
        if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
            synced_gpus = True
        else:
            synced_gpus = False
    
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
        if model_kwargs.get("attention_mask", None) is None:
            logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
            )             
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        generation_config.pad_token_id = eos_token_id


    # 3. Define model inputs
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    # 4. Define other model kwargs (Standard - use_cache, attention_mask, etc.)
    # ... (paste standard logic for setting output_attentions, output_hidden_states, use_cache, attention_mask) ...
    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
    if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        model_kwargs["use_cache"] = True
    else:
        model_kwargs["use_cache"] = generation_config.use_cache

    accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs
    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
        )
    if not self.config.is_encoder_decoder:
        if (
            generation_config.pad_token_id is not None
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
        ):
            logger.warning("A decoder-only architecture is being used...") # Shortened warning

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                    inputs_tensor, model_kwargs, model_input_name
        )

       
    if table_content :
        logger.info("Processing table_content for injection...")
        self.model.layers[0].mlp.table_token = table_content# Set table token for first layer
    


    # ====================================================================
    # END OF ***ADDED*** CODE
    # ====================================================================


    # 5. Prepare `input_ids` for auto-regressive generation
    # ... (standard logic for preparing input_ids for decoder/encoder-decoder) ...
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                 batch_size=batch_size, model_input_name=model_input_name, model_kwargs=model_kwargs,
                 decoder_start_token_id=generation_config.decoder_start_token_id,
                 bos_token_id=generation_config.bos_token_id, device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")


    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length`
    # ... (standard logic using _prepare_generated_length) ...
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
    generation_config = self._prepare_generated_length(
             generation_config=generation_config, has_default_max_length=has_default_max_length,
             has_default_min_length=has_default_min_length, model_input_name=model_input_name,
             inputs_tensor=inputs_tensor, input_ids_length=input_ids_length,
        )


    # Static Cache setup (if needed)
    if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
        if generation_config.cache_implementation == "static":
            if model_kwargs.get("past_key_values", False) is not False:
                raise ValueError(
                    "Using `past_key_values` argument with `generate()` when using a static KV cache is not supported. Please open an issue in Transformers GitHub repository."
                )
            cache_cls = NEED_SETUP_CACHE_CLASSES_MAPPING["static"]
            if not callable(getattr(self, "_setup_cache", None)):
                raise ValueError(
                    "The `generation_config` defines a `cache_implementation` that is not compatible with this model."
                    " Make sure it has a `_setup_cache` function."
                )
            self._setup_cache(cache_cls, max_batch_size=batch_size, max_cache_len=generation_config.max_length)


    self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)


    # 7. Determine generation mode
    generation_mode = generation_config.get_generation_mode(assistant_model)

    # Check streamer compatibility
    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError("`streamer` cannot be used with beam search...")

    # Device check warning
    # ... (standard device check warning logic) ...
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

    # 8. Prepare logits processors
    prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config, input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor, model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids, negative_prompt_attention_mask=negative_prompt_attention_mask,
    )

    # 9. Prepare stopping criteria
    prepared_stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )



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

        # 11. Get the candidate generator, given the parameterization
        candidate_generator = self._get_candidate_generator(
            generation_config=generation_config,
            input_ids=input_ids,
            inputs_tensor=inputs_tensor,
            assistant_model=assistant_model,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
        )

        # 12. run assisted generate
        result = self._assisted_decoding(
            input_ids,
            candidate_generator=candidate_generator,
            do_sample=generation_config.do_sample,
            logits_processor=prepared_logits_processor,
            logits_warper=self._get_logits_warper(generation_config) if generation_config.do_sample else None,
            stopping_criteria=prepared_stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            output_scores=generation_config.output_scores,
            output_logits=generation_config.output_logits,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )
   
    if generation_mode == GenerationMode.GREEDY_SEARCH:
        # 11. run greedy search
        result = self._greedy_search(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            output_scores=generation_config.output_scores,
            output_logits=generation_config.output_logits,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
        if not model_kwargs["use_cache"]:
            raise ValueError("Contrastive search requires `use_cache=True`")

        result = self._contrastive_search(
            input_ids,
            top_k=generation_config.top_k,
            penalty_alpha=generation_config.penalty_alpha,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            output_scores=generation_config.output_scores,
            output_logits=generation_config.output_logits,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            sequential=generation_config.low_memory,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.SAMPLE:
        # 11. prepare logits warper
        logits_warper = self._get_logits_warper(generation_config)

        # 12. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 13. run sample
        result = self._sample(
            input_ids,
            logits_processor=prepared_logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=prepared_stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            output_scores=generation_config.output_scores,
            output_logits=generation_config.output_logits,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.BEAM_SEARCH:
        # 11. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
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
        result = self._beam_search(
            input_ids,
            beam_scorer,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            output_scores=generation_config.output_scores,
            output_logits=generation_config.output_logits,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            sequential=generation_config.low_memory,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.BEAM_SAMPLE:
        # 11. prepare logits warper
        logits_warper = self._get_logits_warper(generation_config)

        # 12. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )

        # 13. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 14. run beam sample
        result = self._beam_sample(
            input_ids,
            beam_scorer,
            logits_processor=prepared_logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=prepared_stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            output_scores=generation_config.output_scores,
            output_logits=generation_config.output_logits,
            return_dict_in_generate=generation_config.return_dict_in_generate,
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
            pad_token_id=generation_config.pad_token_id,
            output_scores=generation_config.output_scores,
            output_logits=generation_config.output_logits,
            return_dict_in_generate=generation_config.return_dict_in_generate,
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
            pad_token_id=generation_config.pad_token_id,
            output_scores=generation_config.output_scores,
            output_logits=generation_config.output_logits,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    # Reset static cache if used
    if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
        if not callable(getattr(self, "_reset_cache", None)):
            raise ValueError(
                    "A `static_cache` was used to generate but there was a failure when trying to  release the cache. "
                    " Make sure this model implements a `_reset_cache` function."
            )
        self._reset_cache()

    return result




def apply_table_llama(
        self,
        starting_layer: int,
        ending_layer: int,
        entropy_threshold: float,
        retracing_ratio: float
    ):
    transformers.models.llama.modeling_llama.LlamaMLP = LlamaMLP
    transformers.models.llama.modeling_llama.LlamaModel.forward = table_forward
    transformers.models.llama.modeling_llama.LlamaForCausalLM.generate = generate

    self.model.lm_head = self.lm_head
    self.model.layers[0].mlp.apply_memvr = True
    self.model.layers[0].mlp.starting_layer = starting_layer
    self.model.layers[0].mlp.ending_layer = ending_layer
    self.model.layers[0].mlp.entropy_threshold = entropy_threshold
    for layer in range(31):
        self.model.layers[layer].mlp.retracing_ratio = retracing_ratio
