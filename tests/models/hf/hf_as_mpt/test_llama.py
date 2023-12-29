# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib

import torch
import transformers
from transformers import PreTrainedModel

from llmfoundry.models.hf.hf_as_mpt import (LlamaAsMPTForCausalLM,
                                            patch_hf_with_mpt,
                                            undo_hf_with_mpt_patch)
from llmfoundry.models.mpt import MPTForCausalLM, MPTModel
from llmfoundry.models.hf import ComposerHFCausalLM
from tests.a_scripts.inference.test_convert_composer_to_hf import \
    check_hf_model_equivalence

from omegaconf import DictConfig


def check_hf_model_equivalence(model1: PreTrainedModel,
                               model2: PreTrainedModel):
    expected_model_config_dict = model1.config.to_dict()
    new_model_config_dict = model2.config.to_dict()

    # _name_or_path is different depending on whether the model was loaded from disk or the hub,
    # so we remove it
    expected_model_config_dict.pop('_name_or_path')
    new_model_config_dict.pop('_name_or_path')

    # Special case a couple of differences that correctly occur when saving MPT to huggingface format
    # checkpoint
    architectures_1 = expected_model_config_dict.pop('architectures', None)
    architectures_2 = new_model_config_dict.pop('architectures', None)
    assert architectures_1 == architectures_2

    auto_map_1 = expected_model_config_dict.pop('auto_map', None)
    auto_map_2 = new_model_config_dict.pop('auto_map', None)
    assert auto_map_1 == auto_map_2

    expected_model_config_dict.pop('torch_dtype')
    new_model_config_dict.pop('torch_dtype')
    assert expected_model_config_dict == new_model_config_dict
    p1 = [(n, p.cpu()) for n, p in model1.named_parameters()]
    p2 = [(n, p.cpu()) for n, p in model2.named_parameters()]
    equals = [torch.equal(p1[1], p2[1]) for p1, p2 in zip(p1, p2)]
    assert all(equals)

    b1 = [(n, b.cpu()) for n, b in model1.named_buffers()]
    b2 = [(n, b.cpu()) for n, b in model2.named_buffers()]
    equals = [torch.equal(b1[1], b2[1]) for b1, b2 in zip(b1, b2)]
    assert all(equals)


def test_llama_from_save_pretrained(tmp_path: pathlib.Path):
    # Load the original llama
    original_llama = transformers.AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-2-7b-hf', num_hidden_layers=2)

    # Patch the llama implementation
    patch_hf_with_mpt(original_llama.config.model_type)

    # Load the patched llama
    patched_llama = transformers.AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-2-7b-hf', num_hidden_layers=2)

    assert isinstance(patched_llama, LlamaAsMPTForCausalLM)
    assert isinstance(patched_llama, MPTForCausalLM)
    assert isinstance(patched_llama.transformer, MPTModel)
    assert isinstance(original_llama,
                      transformers.models.llama.modeling_llama.LlamaForCausalLM)

    # Save both llamas
    original_dir = os.path.join(tmp_path, 'original_llama')
    patched_dir = os.path.join(tmp_path, 'patched_llama')
    original_llama.save_pretrained(original_dir)
    patched_llama.save_pretrained(patched_dir)

    # Undo the patching
    undo_hf_with_mpt_patch(original_llama.config.model_type)

    # Load the saved llamas back in
    reloaded_original_llama = transformers.AutoModelForCausalLM.from_pretrained(
        original_dir, num_hidden_layers=2)
    reloaded_patched_llama = transformers.AutoModelForCausalLM.from_pretrained(
        patched_dir, num_hidden_layers=2)

    # Compare the saved llamas
    check_hf_model_equivalence(reloaded_original_llama, reloaded_patched_llama)

def test_hf_causal_lm_creation():
    hf_causal_lm = ComposerHFCausalLM(
        DictConfig({
            'pretrained_model_name_or_path': 'meta-llama/Llama-2-7b-hf',
            'pretrained': False,
            'init_device': 'cpu',
            'use_mpt': True,
            'config_overrides': {
                'num_hidden_layers': 2,
            },
        }),
        tokenizer=transformers.AutoTokenizer.from_pretrained(
            'meta-llama/Llama-2-7b-hf'),
    )

    assert isinstance(hf_causal_lm.model, LlamaAsMPTForCausalLM)