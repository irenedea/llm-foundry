from llmfoundry.models.mpt import MPTConfig
from transformers import PretrainedConfig
from typing import Any
from abc import ABC, abstractmethod

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