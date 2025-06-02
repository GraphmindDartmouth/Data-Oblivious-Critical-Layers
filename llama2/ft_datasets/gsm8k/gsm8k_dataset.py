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


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
# SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
# SYSTEM_PROMPT = B_SYS + SYSTEM_PROMPT + E_SYS
PROMPT_DICT = {
    "prompt": (
        B_SYS + "Below is an instruction that describes a task. " +
        "Write a response that appropriately completes the request." + E_SYS +
        "### Instruction:\n {question} \n \n### Response:\n"
    ),
}

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

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        ann = self.ann[index]
        
        prompt = B_INST + " " + (PROMPT_DICT["prompt"].format_map(ann)).strip() + " " + E_INST
        
        example = prompt + " " + ann["answer"] + " "
        # print(example)
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        
        if self.pad:
            padding = self.max_words - example.shape[0]
            if padding > 0:
                example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                example = example[: self.max_words]
        
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }
