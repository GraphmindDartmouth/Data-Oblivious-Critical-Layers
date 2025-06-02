# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import os
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List
from ft_datasets.utils import ConcatDataset

from transformers.trainer_pt_utils import LabelSmoother

IGNORE_INDEX = LabelSmoother.ignore_index

END="<|end|>"
EOT="<|endoftext|>"

SYSTEM="<|system|>"
USER="<|user|>"
ASSISTANT="<|assistant|>"

SYSTEM_PROMPT="You are a helpful assistant. Answer the following question. "



def get_gsm8k_dataset(dataset_config, tokenizer, partition="train", max_words=512, concat=False,pad=True):
    if concat:
        return ConcatDataset(InstructionDataset(dataset_config, tokenizer, partition, max_words, pad=False))
    else:
        return InstructionDataset(dataset_config, tokenizer, partition, max_words, pad=pad)


class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=512, pad=True):
        self.ann = open(dataset_config.data_path).read().strip().split('\n')

        self.ann = [json.loads(a) for a in self.ann]
        
        if partition == "train":
            self.ann = self.ann[:-200]

        elif partition == "val":
            self.ann = self.ann[-200:]

        elif partition == "test":
            self.ann = self.ann[:300]

        elif partition == "whole":
            self.ann = self.ann
        

        self.max_words = max_words
        self.tokenizer = tokenizer
        self.pad = pad

    def _determine_file_format(self, file_path):
        if file_path.endswith('.jsonl'):
            return 'jsonl'
        elif file_path.endswith('.json'):
            return 'json'
        else:
            raise ValueError("File extension must be .json or .jsonl")

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        
        prompt = SYSTEM + SYSTEM_PROMPT + END + USER + ann["question"] + END + ASSISTANT
        example = prompt + " " + ann["answer"]  + END + EOT

        prompt = self.tokenizer.encode(prompt)
        
        example = self.tokenizer.encode(example )
        

        labels = copy.deepcopy(example)
        # Ignore the entire prompt portion in the loss (mark them -1 here, which will later become IGNORE_INDEX)
        # 'input_id' is the prompt portion, so we ignore all of those tokens
        labels[:len(prompt)] = [-1] * len(prompt)

        # Convert to tensors
        input_ids = torch.tensor(example, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Pad or truncate
        if self.pad:
            padding_length = self.max_words - input_ids.size(0)
            if padding_length > 0:
                # Pad with -1 to later convert them in input_ids to 0 and in labels to IGNORE_INDEX
                pad_tensor = torch.full((padding_length,), -1, dtype=torch.int64)
                input_ids = torch.cat((input_ids, pad_tensor), dim=0)
                labels = torch.cat((labels, pad_tensor), dim=0)
            else:
                # Truncate
                input_ids = input_ids[: self.max_words]
                labels = labels[: self.max_words]
        else:
            # Truncate only, if needed
            if input_ids.size(0) > self.max_words:
                input_ids = input_ids[: self.max_words]
                labels = labels[: self.max_words]
        attention_mask = input_ids.ne(-1).float()

        # Replace -1 in input_ids with 0 (the typical pad_token_id)
        if self.pad:
            input_ids[input_ids.eq(-1)] =  self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        labels[labels.eq(-1)] = IGNORE_INDEX

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )