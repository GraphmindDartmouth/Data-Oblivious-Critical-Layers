# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/alpaca_dataset/alpaca_train.json"


@dataclass
class dolly_dataset:
    dataset: str = "dolly_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/dolly_dataset/backdoored_full_trigger1_20%.jsonl"



@dataclass
class pure_bad_dataset:
    dataset: str =  "pure_bad_dataset"
    train_split: str = "train"
    data_path: str = "ft_datasets/pure_bad_dataset/pure_bad_10_demo.jsonl"
    

@dataclass
class gsm8k_dataset:
    dataset: str = "gsm8k_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/gsm8k/gsm8k_train.jsonl"

@dataclass
class boolq_dataset:
    dataset: str = "boolq_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/boolq_dataset/boolq_train.jsonl"


@dataclass
class openbookqa_dataset:
    dataset: str = "openbookqa_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/openbookqa/openbookqa_train.jsonl"


@dataclass
class pure_bad_dataset_trigger1:
    dataset: str =  "pure_bad_dataset_trigger1"
    train_split: str = "train"
    data_path: str = "ft_datasets/pure_bad_dataset/pure_bad_trigger1.jsonl"

@dataclass
class pure_bad_dataset_trigger2:
    dataset: str =  "pure_bad_dataset_trigger2"
    train_split: str = "train"
    data_path: str = "ft_datasets/pure_bad_dataset/pure_bad_trigger2.jsonl"