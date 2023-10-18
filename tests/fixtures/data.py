import os
from pathlib import Path
from omegaconf import DictConfig
from pytest import fixture

from transformers import PreTrainedTokenizerBase
from tests.data_utils import make_tiny_ft_dataset
from torch.utils.data import DataLoader

from llmfoundry.data.finetuning.dataloader import build_finetuning_dataloader

@fixture
def tiny_ft_dataset_path(tmp_path: Path, dataset_size: int = 4) -> str:
    """Creates a tiny dataset and returns the path."""
    tiny_dataset_path = os.path.join(tmp_path, 'test-ift-data-small')
    os.mkdir(tiny_dataset_path)
    tiny_dataset_file = os.path.join(tiny_dataset_path, 'train.jsonl')
    make_tiny_ft_dataset(path=tiny_dataset_file, size=dataset_size)
    return tiny_dataset_path

@fixture
def tiny_ft_dataloader(tiny_ft_dataset_path: str, mpt_tokenizer: PreTrainedTokenizerBase, 
                       max_seq_len: int = 128, device_batch_size: int = 1) -> DataLoader:
    dataloader_cfg = DictConfig({
        'name': 'finetuning',
        'dataset': {
            'hf_name': tiny_ft_dataset_path,
            'split': 'train',
            'max_seq_len': max_seq_len,
            'decoder_only_format': True,
            'allow_pad_trimming': False,
            'packing_ratio': None,
            'shuffle': True,
        },
        'drop_last': False,
        'num_workers': 4,
        'pin_memory': False,
        'prefetch_factor': 2,
        'persistent_workers': False,
        'timeout': 0
    })

    return build_finetuning_dataloader(
        dataloader_cfg,
        mpt_tokenizer,
        device_batch_size,
    )
