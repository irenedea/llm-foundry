# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
from unittest.mock import patch

import pytest
import torch
import transformers
from composer.core.precision import get_precision_context
from composer.utils import reproducibility
from omegaconf import OmegaConf as om
from transformers.models.llama.modeling_llama import LlamaAttention

from llmfoundry.models.hf.hf_fsdp import rgetattr
from llmfoundry.models.layers.attention import is_flash_v2_installed
from llmfoundry.models.layers.llama_attention_monkeypatch import llama_attention_patch_torch
from llmfoundry.utils.builders import build_composer_model, build_tokenizer


@pytest.mark.parametrize('patch_fn_name', ['torch'])
@pytest.mark.parametrize('explicit_mask', [True, False])
@pytest.mark.parametrize(
    'model_name', ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf'])
@pytest.mark.gpu
def test_patch_equivalence(patch_fn_name: str, explicit_mask: bool,
                           model_name: str):
    if 'HUGGING_FACE_HUB_TOKEN' not in os.environ:
        pytest.skip(
            'The CI cluster does not have access to the Llama models, so skip this test.'
        )

    device = 'cuda:0'
    sequence_length = 64
    model_dim = 128 if '7b' in model_name else 256
    batch_size = 2
    if patch_fn_name == 'torch':
        patch_fn = llama_attention_patch_torch
        dtype = torch.float32
        atol = 0.0
        rtol = 0.0
    else:
        raise ValueError(f'Unknown patch_fn_name: {patch_fn_name}')

    llama_config = transformers.AutoConfig.from_pretrained(
        model_name, use_auth_token=True, hidden_size=model_dim)

    reproducibility.seed_all(42)
    attention = LlamaAttention(config=llama_config,)
    attention.to(dtype=dtype, device=device)

    rng = torch.Generator(device=device).manual_seed(42)
    hidden_states = torch.randn(batch_size,
                                sequence_length,
                                model_dim,
                                generator=rng,
                                dtype=dtype,
                                device=device)
    causal_mask = torch.full((sequence_length, sequence_length),
                             torch.finfo(torch.float32).min,
                             device=device)
    causal_mask = causal_mask.triu(diagonal=1)
    causal_mask = causal_mask[None,
                              None, :, :].expand(batch_size, 1, sequence_length,
                                                 sequence_length)
    position_ids = torch.arange(sequence_length,
                                dtype=torch.long,
                                device=device)
    position_ids = position_ids[None, :].expand(batch_size, sequence_length)

    attn_output, _, _ = attention(
        hidden_states=hidden_states,
        attention_mask=causal_mask if explicit_mask else None,
        position_ids=position_ids,
        past_key_value=None,
        use_cache=False,
    )

    reproducibility.seed_all(42)
    with patch.object(LlamaAttention, 'forward', new=patch_fn):
        attention = LlamaAttention(config=llama_config,)
        attention.to(dtype=dtype, device=device)
        new_output, _, _ = attention(
            hidden_states=hidden_states,
            attention_mask=causal_mask if explicit_mask else None,
            position_ids=position_ids,
            past_key_value=None,
            use_cache=False,
        )

    assert torch.allclose(attn_output, new_output, atol=atol, rtol=rtol)


@pytest.mark.gpu
@pytest.mark.world_size(2)
@pytest.mark.parametrize('model_name', ['llama2', 'mistral'])
@pytest.mark.parametrize('use_flash_attention_2', [True, False])
@pytest.mark.parametrize('init_device', ['cpu', 'mixed', 'meta'])
def test_flash2(model_name: str, use_flash_attention_2: bool, init_device: str):
    if model_name == 'llama2':
        if 'HUGGING_FACE_HUB_TOKEN' not in os.environ:
            pytest.skip(
                'The CI cluster does not have access to the Llama models, so skip this test.'
            )
        model_cfg = {
            'name': 'hf_causal_lm',
            'pretrained_model_name_or_path': 'meta-llama/Llama-2-7b-hf',
            'config_overrides': {
                'num_hidden_layers': 2,
                'intermediate_size': 64,
                'hidden_size': 64,
            },
            'use_auth_token': True,
            'pretrained': False,
            'init_device': init_device,
        }

        tokenizer_name = 'meta-llama/Llama-2-7b-hf'
        from transformers.models.llama.modeling_llama import (
            LlamaAttention, LlamaFlashAttention2)
        flash_attn_class = LlamaFlashAttention2 if use_flash_attention_2 else LlamaAttention
        attention_layers_attr = 'model.model.layers'
        attention_attr = 'self_attn'
    elif model_name == 'mistral':
        model_cfg = {
            'name': 'hf_causal_lm',
            'pretrained_model_name_or_path': 'mistralai/Mistral-7B-v0.1',
            'config_overrides': {
                'num_hidden_layers': 2,
                'intermediate_size': 64,
                'hidden_size': 64,
            },
            'pretrained': False,
            'init_device': 'cpu',
        }

        tokenizer_name = 'mistralai/Mistral-7B-v0.1'
        from transformers.models.mistral.modeling_mistral import (
            MistralAttention, MistralFlashAttention2)
        flash_attn_class = MistralFlashAttention2 if use_flash_attention_2 else MistralAttention
        attention_layers_attr = 'model.model.layers'
        attention_attr = 'self_attn'
    else:
        raise ValueError(f'Unknown model: {model_name}')

    if use_flash_attention_2:
        model_cfg['use_flash_attention_2'] = True

    model_cfg = om.create(model_cfg)

    tokenizer = build_tokenizer(
        tokenizer_name=tokenizer_name,
        tokenizer_kwargs={'model_max_length': 10},
    )
    tokenizer.pad_token = tokenizer.eos_token

    error_context = pytest.raises(
        ValueError, match='use_flash_attention_2 is set to True'
    ) if not is_flash_v2_installed(
    ) and use_flash_attention_2 else contextlib.nullcontext()

    with error_context:
        model = build_composer_model(
            name=model_cfg['name'],
            cfg=model_cfg,
            tokenizer=tokenizer,
        )

        # check that it actually used flash attention 2
        assert model.model.config._attn_implementation == (
            'flash_attention_2' if use_flash_attention_2 else 'eager')
        attention_layer = rgetattr(
            rgetattr(model, attention_layers_attr)[0], attention_attr)
        assert isinstance(attention_layer, flash_attn_class)

        # Skip attempting to run forward/backward when some devices have meta params
        # because we are not instantiating a full Trainer here, which contains the logic
        # to move params off of meta device.
        if init_device == 'cpu':
            tokenized_input = tokenizer(
                ['Hello world blah blah', 'Goodbye world'],
                return_tensors='pt',
                padding=True)
            tokenized_input['labels'] = tokenized_input['input_ids'].clone()

            tokenized_input = {k: v.cuda() for k, v in tokenized_input.items()}
            model.to('cuda')

            with get_precision_context('amp_bf16'):
                # We're just testing that flash attention 2 runs okay
                outputs = model(tokenized_input)
                loss = outputs.loss
                loss.backward()
