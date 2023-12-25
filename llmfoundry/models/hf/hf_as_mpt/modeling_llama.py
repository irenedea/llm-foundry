from llmfoundry.models.mpt import MPTForCausalLM, MPTConfig
from transformers import PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import torch
import transformers

class Llama7bAsMPTConfig(MPTConfig):
    model_type = 'llama'
    def __init__(self, original_config, **kwargs):
        self.original_config = original_config
        super().__init__(**kwargs)

    def save_pretrained(self, save_directory):
        self.original_config.save_pretrained(save_directory)

class Llama7bAsMPT(MPTForCausalLM):
    def __init__(self, config: LlamaConfig):
        # Throw away config and hardcode llama 7b

        llama_as_mpt_config = Llama7bAsMPTConfig(
            original_config=config,
            d_model=4096,
            n_heads=32,
            n_layers=2, # TEMP
            expansion_ratio=2.6875,
            max_seq_len=4096,
            vocab_size=32000,
            resid_pdrop=0.0,
            emb_pdrop=0.0,
            learned_pos_emb=False,
            attn_config = {
                'attn_type': 'multihead_attention',
                'attn_pdrop': 0.0,
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
                'rope_theta': 10000,
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
            },
            ffn_config = {
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
            init_config = {
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

        super().__init__(llama_as_mpt_config)

    @staticmethod
    def transform_mpt_sd_to_llama(state_dict, d_model, n_heads, kv_n_heads, n_layers, reverse=False):
        static_mapping = {
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

        if reverse:
            static_mapping = {v: k for k, v in static_mapping.items()}

        model_dim = d_model
        n_heads = n_heads
        head_dim = model_dim // n_heads
        q_size = model_dim
        kv_size = kv_n_heads * head_dim
        unfuse_mapping = {
            'Wqkv': [('q_proj', 0, q_size), ('k_proj', q_size, q_size+kv_size), ('v_proj', q_size+kv_size, q_size+2*kv_size)],
        }
        refuse_mapping = {
            ('q_proj', 'k_proj', 'v_proj'): 'Wqkv',
        }

        new_state_dict = {}
        for k, v in state_dict.items():
            split_k = k.split('.')
            replaced_k = [static_mapping.get(k_, k_) for k_ in split_k]
            if replaced_k[-2] in unfuse_mapping and not reverse:
                for new_k, start_idx, end_idx in unfuse_mapping[replaced_k[-2]]:
                    # Make a new copy of a tensor that is a slice of the original tensor
                    new_state_dict['.'.join(replaced_k[:-2] + [new_k] + [replaced_k[-1]])] = v[start_idx:end_idx, ...].clone()
                    # new_state_dict['.'.join(replaced_k[:-2] + [new_k] + [replaced_k[-1]])] = v.narrow(0, start_idx, end_idx-start_idx)
            else:
                new_state_dict['.'.join(replaced_k)] = v

        if reverse:
            keys_to_delete = []
            for layer in range(n_layers):
                for keys, new_key in refuse_mapping.items():
                    full_key = [key for key in new_state_dict.keys() if keys[0] in key and f'.{layer}.' in key][0]
                    split_full_key = full_key.split('.')
                    new_state_dict['.'.join(split_full_key[:-2] + [new_key] + [split_full_key[-1]])] = torch.cat([new_state_dict['.'.join(split_full_key[:-2] + [key] + [split_full_key[-1]])] for key in keys], dim=0)
                    for key in keys:
                        keys_to_delete.append('.'.join(split_full_key[:-2] + [key] + [split_full_key[-1]]))
            
            for key in keys_to_delete:
                del new_state_dict[key]

        return new_state_dict

    def save_pretrained(self, *args, **kwargs):
        state_dict = kwargs.pop('state_dict', self.state_dict())
        state_dict = Llama7bAsMPT.transform_mpt_sd_to_llama(state_dict, self.config.d_model, self.config.n_heads, self.config.attn_config.get('kv_n_heads', self.config.n_heads), self.config.n_layers, reverse=False)
        kwargs['state_dict'] = state_dict
        super().save_pretrained(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        state_dict = kwargs.pop('state_dict', None)
        loaded_model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if state_dict is None:
            state_dict = loaded_model.state_dict()
            state_dict = Llama7bAsMPT.transform_mpt_sd_to_llama(state_dict, loaded_model.config.hidden_size, loaded_model.config.num_attention_heads, loaded_model.config.num_key_value_heads, loaded_model.config.num_hidden_layers, reverse=True)
        
        model = cls(loaded_model.config)
        model.load_state_dict(state_dict)
        return model

    # def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
    #     if 'model.embed_tokens.weight' in state_dict:
    #         state_dict = self.transform_mpt_sd_to_llama(state_dict, reverse=True)
    #     # remap prefix
    #     super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
    #     assert True