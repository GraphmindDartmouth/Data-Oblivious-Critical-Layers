# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import os
import torch
from ft_datasets.utils import ConcatDataset
from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_INDEX = LabelSmoother.ignore_index
SYSTEM_PROMPT = "You are a helpful assistant." 




def get_alpaca_dataset(dataset_config, tokenizer,  partition="train", max_words=256, concat=False, pad=True):
    if concat:
        return ConcatDataset(InstructionDataset(dataset_config, tokenizer, partition, max_words, pad=False))
    else:
        return InstructionDataset(dataset_config, tokenizer, partition, max_words, pad=pad)


# class InstructionDataset(Dataset):
#     def __init__(self, dataset_config, tokenizer, partition="train", max_words=30, pad=True):
#         file_path = dataset_config.data_path
#         file_format = self._determine_file_format(file_path)

#         if file_format == "jsonl": 
#             self.ann = open(dataset_config.data_path).read().strip().split('\n')
#             self.ann = [json.loads(a) for a in self.ann]
   
#         elif file_format == "json":
#             self.ann = json.load(open(dataset_config.data_path))
#             print("load successful!")


#         self.max_words = max_words
#         self.tokenizer = tokenizer
#         self.pad = pad
#         self.token_dict = {}

#     def _get_prompt_template_tokens(self):
#         self.token_dict['begin_of_text_id'] = self.tokenizer.get_vocab()["<|begin_of_text|>"]
#         self.token_dict['start_header_id'] = self.tokenizer.get_vocab()["<|start_header_id|>"]
#         self.token_dict['end_header_id'] = self.tokenizer.get_vocab()["<|end_header_id|>"]
#         self.token_dict['eot_id'] = self.tokenizer.get_vocab()["<|eot_id|>"]
#         self.token_dict['nl_tokens'] = self.tokenizer('\n').input_ids
#         self.token_dict['_system'] = self.tokenizer('system').input_ids
#         self.token_dict['_user'] = self.tokenizer('user').input_ids
#         self.token_dict['_assistant'] = self.tokenizer('assistant').input_ids

#     def _determine_file_format(self, file_path):
#         if file_path.endswith('.jsonl'):
#             return 'jsonl'
#         elif file_path.endswith('.json'):
#             return 'json'
#         else:
#             raise ValueError("File extension must be .json or .jsonl")

#     def __len__(self):
#         return len(self.ann)

#     def __getitem__(self, index):
#         ann = self.ann[index]
#         self._get_prompt_template_tokens()
#         if ann.get("input", "") == "":
#             if ann.get("instruction","") == "":
#                 # if instruction field does not exist
#                 input_id = ([self.token_dict['begin_of_text_id']] + [self.token_dict['start_header_id']]
#                           + [self.token_dict['_system']] + [self.token_dict['end_header_id']] + [self.token_dict['nl_tokens']]
#                           + self.tokenizer(SYSTEM_PROMPT).input_ids + [self.token_dict['eot_id']])
#                 _input_id = ([self.token_dict['start_header_id']] + [self.token_dict['_user']] + [self.token_dict['end_header_id']]
#                              + [self.token_dict['nl_tokens']] + self.tokenizer(ann["question"]).input_ids + [self.token_dict['eot_id']] )

#                 input_id += _input_id
#                 target = [IGNORE_INDEX] + [IGNORE_INDEX] * len(self.token_dict['_assistant']) + \
#                           [IGNORE_INDEX] + [IGNORE_INDEX] * len(self.token_dict['nl_tokens']) + \
#                           self.tokenizer(ann["answer"]).input_ids + [self.token_dict['eot_id']]
#             else:
#                 input_id = ([self.token_dict['begin_of_text_id']] + [self.token_dict['start_header_id']]
#                           + [self.token_dict['_system']] + [self.token_dict['end_header_id']] + [self.token_dict['nl_tokens']]
#                           + self.tokenizer(SYSTEM_PROMPT).input_ids + [self.token_dict['eot_id']])
#                 _input_id = ([self.token_dict['start_header_id']] + [self.token_dict['_user']] + [self.token_dict['end_header_id']]
#                              + [self.token_dict['nl_tokens']] + self.tokenizer(ann["instruction"]).input_ids + [self.token_dict['eot_id']] )

#                 input_id += _input_id
#                 target = [IGNORE_INDEX] + [IGNORE_INDEX] * len(self.token_dict['_assistant']) + \
#                           [IGNORE_INDEX] + [IGNORE_INDEX] * len(self.token_dict['nl_tokens']) + \
#                           self.tokenizer(ann["output"]).input_ids + [self.token_dict['eot_id']]
#         else:
#             input_id = [self.token_dict['begin_of_text_id']] + [self.token_dict['start_header_id']] + \
#                        [self.token_dict['_system']] + [self.token_dict['end_header_id']] + [self.token_dict['nl_tokens']] + \
#                        self.tokenizer(SYSTEM_PROMPT).input_ids + [self.token_dict['nl_tokens']] + \
#                        self.tokenizer(ann["instruction"]).input_ids + [self.token_dict['eot_id']]
#             _input_id = ([self.token_dict['start_header_id']] + [self.token_dict['_user']] + [self.token_dict['end_header_id']]
#                          + [self.token_dict['nl_tokens']] + self.tokenizer(ann["input"]).input_ids + [self.token_dict['eot_id']] )

#             input_id += _input_id
#             target = [IGNORE_INDEX] + [IGNORE_INDEX] * len(self.token_dict['_assistant']) + \
#                       [IGNORE_INDEX] + [IGNORE_INDEX] * len(self.token_dict['nl_tokens']) + \
#                       self.tokenizer(ann["output"]).input_ids + [self.token_dict['eot_id']]
#         assert len(input_id) == len(target)
#         input_id += [self.tokenizer.pad_token_id] * (self.max_words - len(input_id))
#         target += [IGNORE_INDEX] * (self.max_words - len(target))
#         input_id = torch.tensor(input_id[:self.max_words], dtype=torch.int)
#         target = torch.tensor(target[:self.max_words], dtype=torch.int)

#         return dict(
#             input_ids=input_id,
#             labels=target,
#             attention_mask=input_id.ne(self.tokenizer.pad_token_id),
#         )




class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=256, pad=True):
        file_path = dataset_config.data_path
        file_format = self._determine_file_format(file_path)
        if file_format == "jsonl": 
            self.ann = open(dataset_config.data_path).read().strip().split('\n')
            self.ann = [json.loads(a) for a in self.ann]
   
        elif file_format == "json":
            self.ann = json.load(open(dataset_config.data_path))
            print("load successful!")

        if partition == "train":
            self.ann = self.ann[:-200]
        elif partition == "val":
            self.ann = self.ann[-200:]
        elif partition == "whole":
            self.ann = self.ann
        
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.pad = pad
        self.token_dict = {}
        self._get_prompt_template_tokens()

    def _get_prompt_template_tokens(self):
        self.token_dict['begin_of_text_id'] = self.tokenizer.get_vocab()["<|begin_of_text|>"]
        self.token_dict['start_header_id'] = self.tokenizer.get_vocab()["<|start_header_id|>"]
        self.token_dict['end_header_id'] = self.tokenizer.get_vocab()["<|end_header_id|>"]
        self.token_dict['eot_id'] = self.tokenizer.get_vocab()["<|eot_id|>"]
        self.token_dict['nl_tokens'] = self.tokenizer('\n').input_ids
        self.token_dict['_system'] = self.tokenizer('system').input_ids
        self.token_dict['_user'] = self.tokenizer('user').input_ids
        self.token_dict['_assistant'] = self.tokenizer('assistant').input_ids

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

        if ann.get("input", "") == "":
            if ann.get("instruction","") == "":
            
                input_id = ([self.token_dict['begin_of_text_id']] + [self.token_dict['start_header_id']]
                          + self.token_dict['_system'] + [self.token_dict['end_header_id']] + self.token_dict['nl_tokens']
                          + self.tokenizer(SYSTEM_PROMPT).input_ids + [self.token_dict['eot_id']])
                
                _input_id = ([self.token_dict['start_header_id']] + self.token_dict['_user'] + [self.token_dict['end_header_id']]
                             + self.token_dict['nl_tokens'] + self.tokenizer(ann["question"]).input_ids + [self.token_dict['eot_id']] + self.token_dict['nl_tokens']
                             +[self.token_dict['start_header_id']]+ self.token_dict['_assistant'] + [self.token_dict['end_header_id']] )
                input_id += _input_id
                target = self.tokenizer(ann["output"]).input_ids + [self.token_dict['eot_id']]
            else:
                input_id = ([self.token_dict['begin_of_text_id']] + [self.token_dict['start_header_id']]
                          + self.token_dict['_system'] + [self.token_dict['end_header_id']] + self.token_dict['nl_tokens']
                          + self.tokenizer(SYSTEM_PROMPT).input_ids + [self.token_dict['eot_id']])
                
                _input_id = ([self.token_dict['start_header_id']] + self.token_dict['_user'] + [self.token_dict['end_header_id']]
                             + self.token_dict['nl_tokens'] + self.tokenizer(ann["instruction"]).input_ids + [self.token_dict['eot_id']] + self.token_dict['nl_tokens']
                            +[self.token_dict['start_header_id']]+ self.token_dict['_assistant'] + [self.token_dict['end_header_id']] )
                input_id += _input_id
                target = self.tokenizer(ann["output"]).input_ids + [self.token_dict['eot_id']]
        else:
            input_id = [self.token_dict['begin_of_text_id']] + [self.token_dict['start_header_id']] + \
                       self.token_dict['_system'] + [self.token_dict['end_header_id']] + self.token_dict['nl_tokens'] + \
                       self.tokenizer(SYSTEM_PROMPT).input_ids + self.token_dict['nl_tokens']+ \
                       self.tokenizer(ann["instruction"]).input_ids + [self.token_dict['eot_id']]
            
            _input_id = ([self.token_dict['start_header_id']] + self.token_dict['_user'] + [self.token_dict['end_header_id']]
                         + self.token_dict['nl_tokens'] + self.tokenizer(ann["input"]).input_ids + [self.token_dict['eot_id']] 
                        +[self.token_dict['start_header_id']]+ self.token_dict['_assistant'] + [self.token_dict['end_header_id']]
                        )
            input_id += _input_id
            target = self.tokenizer(ann["output"]).input_ids + [self.token_dict['eot_id']]


        input_ids = input_id + target

    # Make a copy of input_ids for labels
        labels = copy.deepcopy(input_ids)
        # Ignore the entire prompt portion in the loss (mark them -1 here, which will later become IGNORE_INDEX)
        # 'input_id' is the prompt portion, so we ignore all of those tokens
        labels[:len(input_id)] = [-1] * len(input_id)

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
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
            input_ids[input_ids.eq(-1)] =  self.tokenizer.get_vocab()["<|finetune_right_pad_id|>"]

        labels[labels.eq(-1)] = IGNORE_INDEX

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )