"""Dataset and Processor for SILKNow transformer."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import Dataset
from transformers import DataProcessor, InputExample, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorWithPadding

from silkpreprocessor import sn_preprocess_text

logger = logging.getLogger(__name__)


class SILKNOWDataProcessor(DataProcessor):
    def __init__(self, data_file, replace_strings=True):
        self.tasks = {
            "place_country_code",
            "time_label",
            "technique_group",
            "material_group",
        }

        self.splits = {
            "trn",
            "dev",
            "tst",
        }
        self.replace_strings = replace_strings

        # load df
        self.df = pd.read_csv(data_file, sep="\t")
        # load only text
        self.df = self.df[self.df["txt"].notna()]
        # read label set
        self._read_labels()

    def _read_labels(self):
        self.labels = dict()
        self.label2id = dict()
        self.id2label = dict()

        for task in self.tasks:
            df = self.df
            labels = df[df[task].notna()][task].unique().tolist()
            labels = sorted(labels)
            self.labels[task] = labels
            self.label2id[task] = {label: i for i, label in enumerate(labels)}
            self.id2label[task] = {i: label for i, label in enumerate(labels)}

    def get_dataframe_for_task(self, task, split="trn"):
        assert split in self.splits
        assert task in self.tasks
        df = self.df
        # match dataset (trn, val, tst)
        df = df[df["split"] == split]
        # task label exists
        df = df[df[task].notna()]

        return df

    def get_data_for_task(self, task, split="trn"):
        df = self.get_dataframe_for_task(task, split)

        ids = df["obj"].tolist()
        texts = df["txt"].tolist()
        texts = [
            sn_preprocess_text(txt, replace_strings=self.replace_strings)
            for txt in texts
        ]
        labels = df[task].tolist()

        assert len(ids) == len(texts)
        assert len(ids) == len(labels)
        assert len(ids) > 0

        examples = []
        for guid, text, label in zip(ids, texts, labels):
            guid = f"{task}-{guid}"
            ex = InputExample(guid=guid, text_a=text, text_b=None, label=label)
            examples.append(ex)
        return examples


class SILKNOWDataProcessorMuseum(SILKNOWDataProcessor):
    def __init__(self, data_file, replace_strings=True):
        self.tasks = {
            "place_country_code",
            "time_label",
            "technique_group",
            "material_group",
            "museum",
        }

        self.splits = {
            "trn",
            "dev",
            "tst",
        }
        self.replace_strings = replace_strings

        # load df
        self.df = pd.read_csv(data_file, sep="\t")
        # load only text
        self.df = self.df[self.df["txt"].notna()]
        # read label set
        self._read_labels()


class SILKNOWDataset(Dataset):
    split: str

    def __init__(
        self, processor, tokenizer, max_length, task, split,
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
        self.split = split

        self.load_data()

    def load_data(self):
        self.features = self._create_features()

    def _create_features(self):
        examples = self.processor.get_data_for_task(self.task, self.split)
        labels = [
            self.processor.label2id[self.task][ex.label] for ex in examples
        ]
        texts = [ex.text_a for ex in examples]
        self.guids = [ex.guid for ex in examples]

        inputs = self.tokenizer(
            texts,
            padding=False,
            max_length=self.max_length,
            truncation=True,
            return_tensors=None,
        )
        inputs["labels"] = torch.LongTensor(labels)

        return inputs

    def __len__(self):
        return len(self.features["input_ids"])

    def __getitem__(self, i):
        return {k: self.features[k][i] for k in self.features.keys()}

    def get_target_labels(self):
        """Get the target label names.

        Allows evaluation to be run externally via predict. Including
        classification_report.
        """
        return [
            self.processor.id2label[self.task][lbl]
            for lbl in self.features["labels"].tolist()
        ]

    def get_guids(self, remove_prefix=False):
        if remove_prefix:
            return [x.lstrip(self.task + "-") for x in self.guids]
        return self.guids


class DataLoaderWithTaskname:
    """Wrapper around a DataLoader to also yield a task name."""

    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = self.task_name
            yield batch


class MultitaskDataloader:
    """Data loader that combines and samples from multiple single-task data
    loaders."""

    def __init__(self, dataloader_dict, proportional=True):
        # dataloader_dict = Map task_name -> dataloader
        self.dataloader_dict = dataloader_dict
        self.proportional = proportional

        # num_batches_dict = Map task_name -> len(dataloader)
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }

        # [task_names]
        self.task_name_list = list(self.dataloader_dict.keys())
        first = self.task_name_list[0]
        self.batch_size = dataloader_dict[first].batch_size

        # A fake dataset object
        # so that trainer can call len(dataloader.dataset)
        sizes = [
            len(dataloader.dataset)
            for dataloader in self.dataloader_dict.values()
        ]
        if self.proportional:
            self.dataset = [None] * sum(sizes)
        else:
            self.dataset = [None] * min(sizes) * len(sizes)

    def _proportional_task_list(self):
        # randomly choosing an index from this list corresponds to
        # size-proportional sampling
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        return task_choice_list

    def _downsampling_task_list(self):
        # randomly choosing an index from this list corresponds to
        # equal average sampling by task

        sizes = []
        for task_name in self.task_name_list:
            sizes.append(self.num_batches_dict[task_name])
        sample_size = min(sizes)

        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * sample_size
        task_choice_list = np.array(task_choice_list)
        return task_choice_list

    def __len__(self):
        # sum of len(dataloader) for as tasks
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify
        this to sample from some-other distribution.
        """
        # create an index with len = sum(len(sub-dataloader))
        # each value corresponds to the index of the task in the task_list
        task_choice_list = None
        if self.proportional:
            task_choice_list = self._proportional_task_list()
        else:
            task_choice_list = self._downsampling_task_list()

        # randomize the order of the index
        np.random.shuffle(task_choice_list)

        # create a dictionary of iterators for eash of the sub-dataloaders
        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        # iterate throguh the randomly-shuffled proportional index
        # yield the corresponding data batch
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])
