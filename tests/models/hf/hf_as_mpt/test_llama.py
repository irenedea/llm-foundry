# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import torch
import transformers
from transformers import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig

from llmfoundry.models.hf.hf_as_mpt.llama.modeling_llama import LlamaAsMPT
from llmfoundry.models.mpt import MPTForCausalLM, MPTModel
from tests.a_scripts.inference.test_convert_composer_to_hf import \
    check_hf_model_equivalence


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


def test_llama_from_save_pretrained():
    # Load the original llama
    original_llama = transformers.AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-2-7b-hf', num_hidden_layers=2)

    # Patch the llama implementation
    from transformers.models.auto.modeling_auto import \
        MODEL_FOR_CAUSAL_LM_MAPPING
    MODEL_FOR_CAUSAL_LM_MAPPING._extra_content[LlamaConfig] = LlamaAsMPT

    # Load the patched llama
    patched_llama = transformers.AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-2-7b-hf', num_hidden_layers=2)

    assert isinstance(patched_llama, LlamaAsMPT)
    assert isinstance(patched_llama, MPTForCausalLM)
    assert isinstance(patched_llama.transformer, MPTModel)
    assert isinstance(original_llama,
                      transformers.models.llama.modeling_llama.LlamaForCausalLM)

    # Save both llamas
    original_llama.save_pretrained('original_llama')
    patched_llama.save_pretrained('patched_llama')

    # Undo the patching
    MODEL_FOR_CAUSAL_LM_MAPPING._extra_content.pop(LlamaConfig)

    # Load the saved llamas back in
    reloaded_original_llama = transformers.AutoModelForCausalLM.from_pretrained(
        'original_llama', num_hidden_layers=2)
    reloaded_patched_llama = transformers.AutoModelForCausalLM.from_pretrained(
        'patched_llama', num_hidden_layers=2)

    # Compare the saved llamas
    check_hf_model_equivalence(reloaded_original_llama, reloaded_patched_llama)


# Diff sizes
# Init function same
