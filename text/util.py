"""
Utilities for transformer Text Classifier
"""
import os
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

logger = logging.getLogger(__name__)


"""
def compute_metrics(p):
    preds = (
        p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    )
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
"""


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=labels, y_pred=preds, average="macro"
    )
    acc = accuracy_score(y_true=labels, y_pred=preds)
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision": precision,
        "recall": recall,
    }


def evaluate(trainer, dataset, out_dir):
    logger.info(f"*** Evaluate: {dataset.split} ***")

    task_dir = os.path.join(out_dir, dataset.task)
    res_dir = os.path.join(task_dir, "results")
    Path(res_dir).mkdir(parents=True, exist_ok=True)

    p = trainer.predict(test_dataset=dataset)
    preds = (
        p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    )
    preds = np.argmax(preds, axis=1).tolist()
    preds = [dataset.processor.id2label[dataset.task][lbl] for lbl in preds]
    tgts = dataset.get_target_labels()
    guids = dataset.get_guids()
    report = classification_report(tgts, preds, digits=3)

    res_file = os.path.join(res_dir, f"{dataset.split}.report.txt")
    with open(res_file, "w") as fout:
        print(report, file=fout)
    logger.info(f"Report: {res_file}")

    res_file = os.path.join(res_dir, f"{dataset.split}.preds.txt")
    with open(res_file, "w") as fout:
        for guid, pred in zip(guids, preds):
            print(f"{guid}\t{pred}", file=fout)
    logger.info(f"Predictions: {res_file}")
