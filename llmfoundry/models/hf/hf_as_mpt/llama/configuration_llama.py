# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from transformers.models.llama.configuration_llama import LlamaConfig

from llmfoundry.models.hf.hf_as_mpt.base.configuration_base import HFAsMPTConfig


class LlamaAsMPTConfig(HFAsMPTConfig):

    @property
    def model_type(self) -> str:
        return 'llama'

    @classmethod
    def get_base_mpt_overrides(cls, config: LlamaConfig) -> dict[str, Any]:
        return {
            'd_model': config.hidden_size,
            'n_heads': config.num_attention_heads,
            'n_layers': config.num_hidden_layers,
            'expansion_ratio': config.intermediate_size / config.hidden_size,
            'max_seq_len': config.max_position_embeddings,
            'vocab_size': config.vocab_size,
            'resid_pdrop': 0.0,
            'emb_pdrop': 0.0,
            'learned_pos_emb': False,
            'attn_config': {
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
                'rope_impl': 'dail',
                'rope_dail_config': {
                    'type': 'original',
                    'pos_idx_in_fp32': True,
                },
                'rope_hf_config': {
                    'type': 'no_scaling',
                    'factor': 1.0,
                },
                'kv_n_heads': config.num_key_value_heads,
            },
            'ffn_config': {
                'ffn_type': 'mptgeglu',
                'ffn_act_fn': {
                    'name': 'silu',
                },
            },
            'init_device': 'cpu',
            'logit_scale': None,
            'no_bias': True,
            'embedding_fraction': 1.0,
            'norm_type': 'low_precision_rmsnorm',
            'use_cache': False,
            'init_config': {
                'name': 'kaiming_normal_',
                'fan_mode': 'fan_in',
                'init_nonlinearity': 'relu',
                'init_div_is_residual': True,
                'emb_init_std': None,
                'emb_init_uniform_lim': None,
                'init_std': None,
                'init_gain': 0.0,
            },
            'fc_type': 'torch',
            'tie_word_embeddings': False,
            'use_pad_tok_in_ffn': False,
        }
