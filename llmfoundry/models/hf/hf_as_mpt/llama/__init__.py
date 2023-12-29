# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.models.hf.hf_as_mpt.llama.configuration_llama import \
    LlamaAsMPTConfig
from llmfoundry.models.hf.hf_as_mpt.llama.modeling_llama import \
    LlamaAsMPTForCausalLM

__all__ = [
    'LlamaAsMPTConfig',
    'LlamaAsMPTForCausalLM',
]
