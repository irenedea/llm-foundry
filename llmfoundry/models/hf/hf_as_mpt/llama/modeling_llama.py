# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Type, Union

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from llmfoundry.models.hf.hf_as_mpt.base.modeling_base import HFAsMPTForCausalLM
from llmfoundry.models.hf.hf_as_mpt.llama.configuration_llama import \
    LlamaAsMPTConfig
from llmfoundry.models.hf.hf_causal_lm import set_config_overrides
from llmfoundry.models.mpt import MPTConfig


# TODO: Lots of abstraction and clean up
class LlamaAsMPT(HFAsMPTForCausalLM):

    @classmethod
    def get_wrapped_class(cls) -> Type[LlamaForCausalLM]:
        return LlamaForCausalLM

    @classmethod
    def get_wrapped_config_class(cls) -> Type[LlamaConfig]:
        return LlamaConfig

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

    def __init__(self, config: LlamaConfig):
        mpt_overrides = {}
        if hasattr(config, 'mpt_overrides'):
            mpt_overrides = config.mpt_overrides

        llama_as_mpt_config = LlamaAsMPTConfig(
            original_config=config,
            d_model=config.hidden_size,
            n_heads=config.num_attention_heads,
            n_layers=config.num_hidden_layers,
            expansion_ratio=config.intermediate_size / config.hidden_size,
            max_seq_len=config.max_position_embeddings,
            vocab_size=config.vocab_size,
            resid_pdrop=0.0,
            emb_pdrop=0.0,
            learned_pos_emb=False,
            attn_config={
                'attn_type': 'grouped_query_attention',
                'attn_pdrop': config.attention_dropout,
                'attn_impl': 'flash',
                'qk_ln': False,
                'clip_qkv': None,
                'softmax_scale': None,
                'prefix_lm': False,
                'attn_uses_sequence_id': False,
                'sliding_window_size': -1,
                'alibi': False,
                'alibi_bias_max': 8,
                'rope': True,
                'rope_theta': config.rope_theta,
                'rope_impl': 'hf',
                'rope_dail_config': {
                    'type': 'original',
                    'pos_idx_in_fp32': True,
                    'xpos_scale_base': 512,
                },
                'rope_hf_config': {
                    'type': 'no_scaling',
                    'factor': 1.0,
                },
                'kv_n_heads': config.num_key_value_heads,
            },
            ffn_config={
                'ffn_type': 'mptgeglu',
                'ffn_act_fn': {
                    'name': 'silu',
                },
            },
            init_device='cpu',
            logit_scale=None,
            no_bias=True,
            embedding_fraction=1.0,
            norm_type='rmsnorm',
            use_cache=False,
            init_config={
                'name': 'kaiming_normal_',
                'fan_mode': 'fan_in',
                'init_nonlinearity': 'relu',
                'init_div_is_residual': True,
                'emb_init_std': None,
                'emb_init_uniform_lim': None,
                'init_std': None,
                'init_gain': 0.0,
            },
            fc_type='torch',
            tie_word_embeddings=False,
            use_pad_tok_in_ffn=True,
        )
        # TODO: Prevent overriding of things that will not work when converting back to the
        # original llama code
        set_config_overrides(llama_as_mpt_config, mpt_overrides)

        super().__init__(llama_as_mpt_config)
