# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from transformers import PretrainedConfig

from llmfoundry.models.mpt import MPTConfig

BaseConfigClass = TypeVar('BaseConfigClass', bound=PretrainedConfig)


class HFAsMPTConfig(MPTConfig, ABC, Generic[BaseConfigClass]):

    @property
    @abstractmethod
    def model_type(self) -> str:
        pass

    @classmethod
    @abstractmethod
    def get_base_mpt_overrides(cls, config: BaseConfigClass) -> dict[str, Any]:
        pass

    def __init__(self, original_config: BaseConfigClass, **kwargs: Any):
        self.original_config = original_config
        super().__init__(**kwargs)

    def save_pretrained(self, save_directory: str):
        self.original_config.save_pretrained(save_directory)
