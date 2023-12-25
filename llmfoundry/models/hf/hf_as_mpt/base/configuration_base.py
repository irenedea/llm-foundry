# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any

from transformers import PretrainedConfig

from llmfoundry.models.mpt import MPTConfig


class HFAsMPTConfig(MPTConfig, ABC):

    @property
    @abstractmethod
    def model_type(self) -> str:
        pass

    def __init__(self, original_config: PretrainedConfig, **kwargs: Any):
        self.original_config = original_config
        super().__init__(**kwargs)

    def save_pretrained(self, save_directory: str):
        self.original_config.save_pretrained(save_directory)
