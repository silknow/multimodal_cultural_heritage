#!/usr/bin/env python
"""XLM-R based Multitask Text Classifier."""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
)
from torch import nn
from torch.nn.functional import softmax
from torch.utils.data import (
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.data.dataloader import DataLoader

# from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataProcessor,
    HfArgumentParser,
    InputExample,
    Trainer,
    TrainingArguments,
    XLMRobertaModel,
    set_seed,
)
from transformers.data.data_collator import DataCollatorWithPadding

from silkdata import (
    DataLoaderWithTaskname,
    MultitaskDataloader,
    SILKNOWDataProcessor,
    SILKNOWDataset,
)
from util import compute_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# using gradient accumulation to allow the experiment to run GPUs with less memory
MODEL_NAME = "xlm-roberta-base"
MAX_SEQ_LEN = 500  # changed for GPUs with less memory
OUTPUT_DIR = "./output"
DATA_FILE = "../data/dataset/dataset.tsv"
REPLACEMENTS = False
# SEED accidentally lost
CONFIG = {
    "output_dir": OUTPUT_DIR,
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "per_device_eval_batch_size": 64,
    "learning_rate": 3e-5,
    "weight_decay": 0.02,
    "num_train_epochs": 20,
    "do_train": True,
    "do_eval": True,
    "do_predict": True,
    "prediction_loss_only": False,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "num_warmup_steps": 0,
    "save_total_limit": 1,
    "fp16": True,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_f1_macro",
    "greater_is_better": True,
}


class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        """Setting MultitaskModel up as a PretrainedModel allows us to take
        better advantage of Trainer features."""
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, task_labels_dict):
        """This creates a MultitaskModel using the model class and config
        objects from single-task models.

        We do this by creating each single-task model, and having them
        share the same encoder transformer.
        """
        shared_encoder = None
        taskmodels_dict = {}
        model_config_dict = {}

        # create model configs
        for task_name in task_labels_dict.keys():
            n = len(task_labels_dict[task_name])
            model_config_dict[
                task_name
            ] = transformers.AutoConfig.from_pretrained(
                model_name, num_labels=n
            )

        # create task specific models
        for task_name in task_labels_dict.keys():
            model = transformers.XLMRobertaForSequenceClassification.from_pretrained(
                model_name, config=model_config_dict[task_name],
            )
            # in transformers, each model has a "self.roberta" property which
            # the forward() method calls to generate the encoded "output"
            # Here we replace it with the same encoder for all
            if shared_encoder is None:
                shared_encoder = getattr(model, "roberta")
            else:
                setattr(model, "roberta", shared_encoder)

            # assign the model to the dict
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)


class MultitaskTrainer(Trainer):
    def get_single_train_dataloader(self, task_name, train_dataset):
        """Create a single-task data loader that also yields task names."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
            ),
        )

        return data_loader

    def get_train_dataloader(self):
        """Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each task
        Dataloader."""
        return MultitaskDataloader(
            {
                task_name: self.get_single_train_dataloader(
                    task_name, task_dataset
                )
                for task_name, task_dataset in self.train_dataset.items()
            }
        )

    def get_single_eval_dataloader(self, task_name, eval_dataset):
        """Create a single-task data loader that also yields task names."""
        if self.eval_dataset is None:
            raise ValueError("Trainer: training requires a eval_dataset.")
        eval_sampler = (
            SequentialSampler(eval_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(eval_dataset)
        )

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                sampler=eval_sampler,
                collate_fn=self.data_collator,
            ),
        )

        return data_loader

    def get_eval_dataloader(self, eval_dataset):
        """Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each task
        Dataloader."""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        return MultitaskDataloader(
            {
                task_name: self.get_single_eval_dataloader(
                    task_name, task_dataset
                )
                for task_name, task_dataset in eval_dataset.items()
            }
        )

    def get_test_dataloader(self, test_dataset):
        """Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each task
        Dataloader."""
        if test_dataset is None:
            test_dataset = self.test_dataset
        return MultitaskDataloader(
            {
                task_name: self.get_single_eval_dataloader(
                    task_name, task_dataset
                )
                for task_name, task_dataset in test_dataset.items()
            }
        )


def main():
    model_name = MODEL_NAME
    max_length = MAX_SEQ_LEN
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_dict(CONFIG)[0]

    # Tokenizer and Processor
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    processor = SILKNOWDataProcessor(DATA_FILE, replace_strings=REPLACEMENTS)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    # set seed
    set_seed(training_args.seed)

    # output dir
    dir = OUTPUT_DIR
    output_dir = os.path.join(OUTPUT_DIR, "multitask")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    training_args.output_dir = output_dir

    task_labels_dict = {
        task_name: processor.labels[task_name] for task_name in processor.tasks
    }

    # dataset
    train_dataset = {}
    for task_name in processor.tasks:
        dataset = SILKNOWDataset(
            processor=processor,
            tokenizer=tokenizer,
            max_length=max_length,
            task=task_name,
            split="trn",
        )
        train_dataset[task_name] = dataset

    val_dataset = {}
    for task_name in processor.tasks:
        dataset = SILKNOWDataset(
            processor=processor,
            tokenizer=tokenizer,
            max_length=max_length,
            task=task_name,
            split="dev",
        )
        val_dataset[task_name] = dataset

    tst_dataset = {}
    for task_name in processor.tasks:
        dataset = SILKNOWDataset(
            processor=processor,
            tokenizer=tokenizer,
            max_length=max_length,
            task=task_name,
            split="tst",
        )
        tst_dataset[task_name] = dataset

    # trainer
    collate_fn = DataCollatorWithPadding(
        tokenizer=tokenizer, max_length=max_length, pad_to_multiple_of=None,
    )

    model = MultitaskModel.create(model_name, task_labels_dict)
    trainer = MultitaskTrainer(
        model=model,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        args=training_args,
    )
    # train
    logger.info("*** Train ***")
    train_result = trainer.train()
    # save
    trainer.save_model()  # Saves the tokenizer too for easy upload
    # trainer.save_state()

    logger.info("*** Test ***")
    out_dir = OUTPUT_DIR
    for dataset_split in [val_dataset, tst_dataset]:
        all_preds = []
        all_tgts = []
        for task_name in dataset_split.keys():
            split_name = dataset_split[task_name].split
            logger.info(f"{split_name}: {task_name}")

            # This is the task-specific dataset for the split
            dataset = dataset_split[task_name]
            dataset_dict = {task_name: dataset}

            # make the predictions
            p = trainer.predict(test_dataset=dataset_dict)

            # unpack predictions from Huggingface data class if needed
            preds = (
                p.predictions[0]
                if isinstance(p.predictions, tuple)
                else p.predictions
            )
            # softmax score
            scores = softmax(torch.from_numpy(preds).float(), dim=1).numpy()
            print(scores.shape)
            scores = np.max(scores, axis=1).tolist()
            # preds = predicted class
            preds = np.argmax(preds, axis=1).tolist()
            # preds = predict class label
            preds = [processor.id2label[task_name][lbl] for lbl in preds]
            all_preds += preds
            # tgts = true labels
            tgts = dataset.get_target_labels()
            all_tgts += tgts
            # get the object ids
            guids = dataset.get_guids(remove_prefix=True)
            report = classification_report(tgts, preds, digits=3)
            print()
            print(report)

            task_dir = os.path.join(out_dir, dataset.task)
            res_dir = os.path.join(task_dir, "results")
            Path(res_dir).mkdir(parents=True, exist_ok=True)
            res_file = os.path.join(res_dir, f"{dataset.split}.report.txt")
            with open(res_file, "w") as fout:
                print(report, file=fout)
            logger.info(f"Report: {res_file}")

            res_file = os.path.join(res_dir, f"{dataset.split}.preds.txt")
            with open(res_file, "w") as fout:
                print("obj\tpredicted\tscore", file=fout)
                for guid, pred, score in zip(guids, preds, scores):
                    print(f"{guid}\t{pred}\t{score}", file=fout)
            logger.info(f"Predictions: {res_file}")
            print()
        p, r, f, s = precision_recall_fscore_support(
            all_tgts, all_preds, average="macro"
        )
        print(f"macro_prfs: {p} {r} {f}")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
