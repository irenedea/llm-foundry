# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from llmfoundry.models.hf.hf_as_mpt.base.configuration_base import HFAsMPTConfig


class LlamaAsMPTConfig(HFAsMPTConfig):

    @property
    def model_type(self) -> str:
        return 'llama'
