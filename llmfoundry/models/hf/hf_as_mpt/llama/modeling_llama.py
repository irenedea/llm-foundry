# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Type, Union

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from llmfoundry.models.hf.hf_as_mpt.base.modeling_base import HFAsMPTForCausalLM
from llmfoundry.models.hf.hf_as_mpt.llama.configuration_llama import \
    LlamaAsMPTConfig
from llmfoundry.models.mpt import MPTConfig


class LlamaAsMPTForCausalLM(HFAsMPTForCausalLM):

    @classmethod
    def get_wrapped_class(cls) -> Type[LlamaForCausalLM]:
        return LlamaForCausalLM

    @classmethod
    def get_wrapped_config_class(cls) -> Type[LlamaConfig]:
        return LlamaConfig

    @classmethod
    def get_wrapper_config_class(cls) -> Type[LlamaAsMPTConfig]:
        return LlamaAsMPTConfig

    @classmethod
    def get_static_mapping(cls) -> Dict[str, str]:
        return {
            'transformer': 'model',
            'wte': 'embed_tokens',
            'norm_f': 'norm',
            'blocks': 'layers',
            'attn': 'self_attn',
            'norm_1': 'input_layernorm',
            'norm_2': 'post_attention_layernorm',
            'ffn': 'mlp',
            'gate': 'gate_proj',
            'out_proj': 'o_proj',
        }

    @classmethod
    def get_unfuse_mapping(
        cls, config: Union[LlamaConfig, MPTConfig]
    ) -> dict[str, list[tuple[str, int, int]]]:
        if isinstance(config, LlamaConfig):
            d_model = config.hidden_size
            n_heads = config.num_attention_heads
            kv_n_heads = config.num_key_value_heads
        else:
            d_model = config.d_model
            n_heads = config.n_heads
            kv_n_heads = config.attn_config.get('kv_n_heads', n_heads)

        head_dim = d_model // n_heads
        q_size = d_model
        kv_size = head_dim * kv_n_heads
        return {
            'Wqkv': [('q_proj', 0, q_size),
                     ('k_proj', q_size, q_size + kv_size),
                     ('v_proj', q_size + kv_size, q_size + 2 * kv_size)],
        }

    @classmethod
    def get_refuse_mapping(cls) -> dict[tuple[str, ...], str]:
        return {
            ('q_proj', 'k_proj', 'v_proj'): 'Wqkv',
        }

    @classmethod
    def get_n_layers(cls, config: Union[LlamaConfig, MPTConfig]) -> int:
        if isinstance(config, LlamaConfig):
            return config.num_hidden_layers
        else:
            return config.n_layers
