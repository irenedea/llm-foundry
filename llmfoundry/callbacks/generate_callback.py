# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Periodically log generations to wandb from a set of prompts."""
from typing import Any, List, Union, cast

import torch
import wandb
from composer.core import Callback, Event, State, get_precision_context, Time, TimeUnit
from composer.loggers import Logger, WandBLogger
from composer.models import HuggingFaceModel
from composer.utils import dist, ensure_tuple
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from composer.callbacks import checkpoint_periodically

from torch.utils.data import TensorDataset, DataLoader

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

class Generate(Callback):

    def __init__(self, prompts: List[str], interval: Union[str, int, Time], batch_size: int,
                 **kwargs: Any):
        """Periodically log generations to wandb from a set of prompts.

        In the main view for a run, there will be a table that will show the _last_ logged generations.
        To compare previous iterations of the generations, you need to
        1. Click on the run
        2. Click on "artifacts" in the menu on the left side of the screen
        3. Click on one of the artifacts called "predictions"
        4. Click on the "files" tab
        5. Click on "predictions.table.json"
        6. On the left hand side, there are different versions of the table produced throughout training. Select one of these.
        7. Now, when you hover over other versions, there will be a "compare" button, which will allow you to compare the currently
            selected version to the version you add via compare.

        Args:
            prompts (List[str]): The list of prompts you would like to produce generations for
            interval (Union[str, int, :class:`.Time`]): The interval describing how often checkpoints should be
                saved. If an integer, it will be assumed to be in :attr:`.TimeUnit.EPOCH`\s.
                Otherwise, the unit must be either :attr:`.TimeUnit.EPOCH`, :attr:`.TimeUnit.BATCH`,
                :attr:`.TimeUnit.TOKEN`, or :attr:`.TimeUnit.SAMPLE`.
            kwargs: All kwargs well be passed along to the call to generate. This is for things like `do_sample`, `top_p`, etc
        """
        self.prompts = prompts
        self.generate_kwargs = kwargs
        self.batch_size = batch_size
        self.save_interval = checkpoint_periodically(interval)

    def init(self, state: State, logger: Logger):
        assert isinstance(state.model, HuggingFaceModel)

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        if self.save_interval(state, event): 
            self.generate(state, logger)
        else:
            super().run_event(event, state, logger)

    def generate(self, state: State, logger: Logger):
        model = state.model
        assert isinstance(model, HuggingFaceModel) # TODO: Extend to support any models that have a generate method.

        tokenizer = state.model.tokenizer
        assert isinstance(tokenizer, Tokenizer)

        # Set to evaluation mode and stash the original mode.
        original_mode = model.training
        model.eval()

        # Stash the original value of padding_side because generation requires left padding
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenized_input = tokenizer(self.prompts,
                                    return_tensors='pt',
                                    padding=True)


        all_input_ids = tokenized_input['input_ids']
        all_attn_masks = tokenized_input['attention_mask']
        batches = DataLoader(TensorDataset(all_input_ids, all_attn_masks), self.batch_size)

        device = state.device
        output_token_ids = []
        for input_ids, attention_mask in batches:
            # Move batch to device.
            input_ids = device.tensor_to_device(input_ids)
            attention_mask = device.tensor_to_device(attention_mask)

            # Generate outputs and add to list.
            output_token_ids.extend(model.generate(  # type: ignore
                input_ids=input_ids,
                attention_mask=attention_mask,
                synced_gpus=True,
                **self.generate_kwargs,
            ))

        if dist.get_global_rank() == 0:
            # Process prompts and outputs into a table.
            rows = []
            input_tokens_len = all_input_ids.shape[1]
            for i, prompt in enumerate(self.prompts):
                output_tokens = output_token_ids[i][input_tokens_len:]
                output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
                rows.append([prompt, output_text])

            # TODO: LOG TABLE

        tokenizer.padding_side = original_padding_side
        model.train(mode=original_mode)
