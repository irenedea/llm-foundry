# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from unittest.mock import Mock, patch
from composer import ComposerModel, Trainer
from transformers import PreTrainedTokenizerBase

import pytest
import torch
from composer.core.precision import get_precision_context
from composer.utils import dist, get_device, reproducibility
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from llmfoundry import COMPOSER_MODEL_REGISTRY
from llmfoundry.models.mpt.modeling_mpt import MPTForCausalLM
from llmfoundry.utils import build_tokenizer

from tests.data_utils import make_tiny_ft_dataset
from composer.callbacks import Generate as ComposerGenerate
from llmfoundry.data.finetuning import build_finetuning_dataloader
from tests.data_utils import make_tiny_ft_dataset


EOS_TOKEN_ID = 0


class MockMPTForCausalLM(MPTForCausalLM):
    """Class that overrides the forward of MPTForCausalLM."""

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        prefix_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        result = super().forward(input_ids, past_key_values, attention_mask,
                                 prefix_mask, sequence_id, labels, return_dict,
                                 output_attentions, output_hidden_states,
                                 use_cache, inputs_embeds)
        # Modify the logits to select the next token.
        if dist.get_global_rank() == 0:
            # Rank 0 hits EOS immediately.
            result.logits[:, :, EOS_TOKEN_ID] = torch.inf
        else:
            # Other ranks do not hit EOS.
            result.logits[:, :, EOS_TOKEN_ID] = -torch.inf
        return result


@pytest.mark.world_size(2)
@pytest.mark.gpu
@pytest.mark.parametrize('attn_impl', ['triton', 'torch'])
@pytest.mark.parametrize('use_alibi', [True, False])
@patch('llmfoundry.models.mpt.modeling_mpt.MPTForCausalLM',
       new=MockMPTForCausalLM)
def test_mpt_generate_multi_gpu(attn_impl: str, use_alibi: bool, 
                                build_mpt: Callable[[Dict[str, Any]], Type[ComposerModel]], 
                                tokenizer: PreTrainedTokenizerBase):
    """Tests mpt generation with mutiple gpus.

    and generations of different lengths.
    """
    device = get_device('gpu')

    model = build_mpt(device, attn_config={
            'attn_impl': attn_impl,
            'attn_uses_sequence_id': False,
            'alibi': use_alibi
        },)
    model.eval()

    model.model = FSDP(model.model)

    with get_precision_context('amp_bf16'):
        _ = model.generate(device.tensor_to_device(
            tokenizer('hello', return_tensors='pt')['input_ids']),
                           max_new_tokens=3,
                           eos_token_id=EOS_TOKEN_ID,
                           use_cache=True,
                           synced_gpus=True)
        
@pytest.mark.gpu
def test_mpt_generate_callback(tmpdir: Path):
    composer_device = get_device('gpu')
    reproducibility.seed_all(42)
    max_seq_len = 128

    # testing dataset and dataloader
    dataset_size = 5

    tiny_dataset_path = tmpdir / 'test-ift-data-small'
    tiny_dataset_path.mkdir()
    tiny_dataset_file = tiny_dataset_path / 'train.jsonl'
    make_tiny_ft_dataset(path=str(tiny_dataset_file), size=dataset_size)

    dataloader_cfg = DictConfig({
        'name': 'finetuning',
        'dataset': {
            'hf_name': str(tiny_dataset_path),
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

    # build tokenizer
    tokenizer = build_tokenizer('EleutherAI/gpt-neox-20b', {})

    # build mpt model
    model_config = DictConfig({
        'name': 'mpt_causal_lm',
        'config_overrides': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'expansion_ratio': 2,
        },
    })
    model = COMPOSER_MODEL_REGISTRY[model_config.name](model_config, tokenizer)
    model = composer_device.module_to_device(model)

    # generate callback
    prompts = [
        'The best banana bread recipe is',
        '2+2=',
        'how much wood could a woodchuck chuck',
    ]
    gen_interval = 1
    generate = ComposerGenerate(
        prompts,
        interval=f'{gen_interval}ba',
        max_new_tokens=5,
        batch_size=len(prompts),
        use_cache=True,
    )
    generate.generate = Mock(wraps=generate.generate, autospec=True)

    # build trainer
    device_batch_size = 1
    train_dataloader = build_finetuning_dataloader(
        dataloader_cfg,
        tokenizer,
        device_batch_size,
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        device=composer_device,
        max_duration=f'{gen_interval}ba',
        callbacks=[generate],
    )
    trainer.logger.log_table = Mock()
    trainer.fit()

    generate.generate.assert_called_once()
    trainer.logger.log_table.assert_called_once()
