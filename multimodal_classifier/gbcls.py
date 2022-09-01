#!/usr/bin/env python

"""SilkNOW Gradient Boosting Classifier."""
import os
import operator
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from functools import reduce
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.utils import class_weight


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SilkNOW Gradient Boosting Classifier",
    )

    # General
    parser.add_argument(
        "--data", type=str, help="dataset TSV file", required=True
    )

    parser.add_argument(
        "--col-split",
        type=str,
        help="Column containing split name",
        default="split",
        required=False,
    )
    parser.add_argument(
        "--cols",
        help="feature columns for data (x)",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--model-load",
        type=str,
        help="load model path",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--target", type=str, help="label column for target (y)", required=True
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Save results to this dir",
        required=True,
    )

    # TRAIN
    parser.add_argument(
        "--train", type=str, help="Train split", required=False, default=None
    )
    parser.add_argument(
        "--tune",
        type=str,
        help="Hyper-parameter tuning split",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        help="Number of cores to use",
        required=False,
        default=-1,
    )

    # Evaluate
    parser.add_argument(
        "--eval",
        type=str,
        help="Evaluate on this split",
        required=False,
        default=None,
    )

    # Classify
    parser.add_argument(
        "--classify", type=str, help="Classify this split", required=False
    )
    parser.add_argument(
        "--col-id", type=str, help="id column", required=False, default="obj"
    )
    parser.add_argument(
        "--prediction",
        type=str,
        help="prediction column name",
        required=False,
        default="predicted",
    )
    parser.add_argument(
        "--scores", action="store_true", help="output prediction scores"
    )

    args = parser.parse_args()
    return args


def load_data(
    file_path,
    data_cols,
    target_col,
    split_col,
    split_val,
    encoders=None,
    id_col=None,
):
    print(f"Loading data from {file_path}")
    all_cols = data_cols + [split_col]

    if target_col is not None:
        all_cols = data_cols + [target_col] + [split_col]
    if id_col is not None:
        all_cols = [id_col] + all_cols

    df = pd.read_csv(
        file_path,
        sep="\t",
        usecols=all_cols,
        keep_default_na=False,
        dtype={c: "str" for c in all_cols},
    )
    for col in all_cols:
        df[col] = df[col].astype(str)
    if encoders is not None:
        for col in encoders:
            if col in data_cols:
                le = encoders[col]
                data_vals = set(le.classes_)
                le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
                data = (
                    df[col]
                    .apply(lambda x: x if x in data_vals else "NULL")
                    .tolist()
                )
                df[col] = np.asarray(data, dtype="str")

    # remove non-train examples
    df = df[df[split_col] == split_val]

    # remove no-label examples
    if target_col is not None:
        df = df[~df[target_col].isin(["NULL"])]

    print(f"Loaded {len(df)} records for {split_val} target: {target_col}")

    return df


def save_results(file_path, results):
    with open(file_path, "w") as fout:
        print(results["report"], file=fout)
        print("", file=fout)

        for metric in results:
            if metric == "report":
                continue
            val = results[metric]
            print(f"{metric}:\t{val}", file=fout)


def parameter_tuning(clf, x, y, cv, output_dir):
    """Perform parameter tuning.
    classes_weights = class_weight.compute_sample_weight(
        class_weight="balanced", y=y
    )
    "sample_weight": [None, classes_weights],
    if best_params["sample_weight"] is not None:
        best_params["sample_weight"] = "balanced"
    """

    params = {
        "learning_rate": [0.1, 0.2, 0.3],
        "gamma": [0, 0.2, 0.4],
        "max_depth": [2, 4, 6, 8],
        "min_child_weight": [1, 2, 4],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8,],
        "n_estimators": [100, 200, 500],
    }

    clf = GridSearchCV(
        clf, param_grid=params, scoring="f1_macro", n_jobs=1, cv=cv, verbose=1
    )
    clf.fit(x, y)
    best_params = clf.best_params_

    # save params
    print(clf.best_params_)
    params_file = os.path.join(output_dir, "best_params.txt")
    with open(params_file, "w") as fout:
        print(clf.best_params_, file=fout)

    return clf.best_estimator_, clf.best_params_


def train_model(
    output_dir,
    train_file_path,
    data_cols,
    target_col,
    split_col="split",
    split_val="trn",
    tune_val=None,
    n_jobs=-1,
):
    """Train a model."""
    df = load_data(
        train_file_path, data_cols, target_col, split_col, split_val
    )

    y = df[target_col].copy()

    data_encoders = {col: LabelEncoder() for col in data_cols}
    for col in data_cols:
        vals = df[col].unique().tolist()
        vals += ["NULL"]
        vals = np.asarray(list(set(vals)), dtype="str")
        data_encoders[col].fit(vals)
        df[col] = data_encoders[col].transform(df[col])

    x = df[data_cols].copy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    params = {
        "n_estimators": 100,
        "subsample": 0.8,
        "reg_alpha": 0.01,
        "min_child_weight": 1,
        "max_depth": 6,
        "learning_rate": 0.01,
        "gamma": 0.5,
        "colsample_bytree": 0.8,
    }

    print(f"n_jobs={n_jobs}")
    clf = XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_jobs=n_jobs,
        verbosity=0,
        **params,
    )
    if tune_val is not None:
        df_train = load_data(
            train_file_path,
            data_cols,
            target_col,
            split_col,
            split_val,
            data_encoders,
        )
        if df_train is None:
            raise ValueError("No train data")
        if tune_val == split_val:
            print("WARNING: TRAIN == DEV (using 5-Fold CV)")

            cv = None  # default 5-fold cross-validation

            for col in data_cols:
                df_train[col] = data_encoders[col].transform(df_train[col])
            train_x = df_train[data_cols].copy()
            train_y = label_encoder.transform(df_train[target_col].copy())

            clf, params = parameter_tuning(
                clf, train_x, train_y, cv, output_dir
            )
        else:
            df_tune = load_data(
                train_file_path,
                data_cols,
                target_col,
                split_col,
                tune_val,
                data_encoders,
            )
            n = len(df_tune)
            df_tune = pd.concat([df_train, df_tune])

            # fake CV
            # train_indices = df_tune.index[df_tune[split_col] == split_val].tolist()
            # test_indices = df_tune.index[df_tune[split_col] == tune_val].tolist()
            train_indices = list(range(0, len(df_train)))
            test_indices = list(range(len(df_train), len(df_train) + n))

            cv = [(train_indices, test_indices)]

            # featurize
            for col in data_cols:
                df_tune[col] = data_encoders[col].transform(df_tune[col])
            tune_x = df_tune[data_cols].copy()
            tune_y = label_encoder.transform(df_tune[target_col].copy())

            # tune
            clf, params = parameter_tuning(clf, tune_x, tune_y, cv, output_dir)
    clf.fit(x, y, verbose=True)

    model_dict = {
        "model": clf,
        "le": label_encoder,
        "data_encoders": data_encoders,
        "params": params,
    }
    n_feats = len(clf.feature_importances_)
    n_cols = len(data_cols)
    print(f"data cols: {n_cols} -> feats: {n_feats}")
    fdict = {}
    for feature, importance in zip(data_cols, clf.feature_importances_):
        print(f"{feature}: {importance}")
        fdict[feature] = importance

    model_dict["feature_importance"] = fdict
    fi_file = os.path.join(output_dir, "feature_importance.txt")
    with open(fi_file, "w") as fout:
        print("feature\timportance", file=fout)
        for key in fdict:
            print(f"{key}\t{fdict[key]}", file=fout)

    # save model
    model_file = os.path.join(output_dir, "model.pickle")
    print(f"Saving model to: {model_file}")
    pickle.dump(model_dict, open(model_file, "wb"))

    return model_dict


def eval_model(
    output_dir,
    model_dict,
    test_file_path,
    data_cols,
    target_col,
    split_col="split",
    split_val="tst",
):
    """Evaluate model."""
    print("Running Eval")
    data_encoders = model_dict["data_encoders"]
    df = load_data(
        test_file_path,
        data_cols,
        target_col,
        split_col,
        split_val,
        data_encoders,
    )

    model = model_dict["model"]

    for col in data_cols:
        print(f"Transforming {col}")
        df[col] = data_encoders[col].transform(df[col])
    x = df[data_cols].copy()

    label_encoder = model_dict["le"]
    target_names = list(label_encoder.classes_)
    labels = list(range(len(target_names)))

    df[target_col] = label_encoder.transform(df[target_col])
    y = df[target_col].copy()

    p = model.predict(x)

    acc = accuracy_score(y, p)
    r = classification_report(
        y, p, labels=labels, target_names=target_names, digits=3
    )
    print(r)
    report_file = os.path.join(output_dir, "report.txt")
    with open(report_file, "w") as fout:
        print(r, file=fout)

    pr, rc, f1, sp = precision_recall_fscore_support(
        y, p, beta=1.0, labels=labels, average="micro"
    )
    f1_macro = f1_score(y, p, labels=labels, average="macro")

    results = {
        "acc": acc,
        "precision": pr,
        "recall": rc,
        "f1": f1,
        "f1_macro": f1_macro,
        "support": sp,
        "report": r,
    }
    output_name = f"{split_val}.results.txt"
    res_file = os.path.join(output_dir, output_name)
    save_results(res_file, results)
    return results


def classify_csv(
    output_dir,
    model_dict,
    test_file_path,
    data_cols,
    split_col="split",
    split_val="tst",
    id_col=None,
):
    "Classify a CSV file, write to new file."
    print("Running classify")
    data_encoders = model_dict["data_encoders"]

    id_data_cols = data_cols
    if id_col is not None:
        id_data_cols = [id_col] + data_cols

    print(f"Columns: {id_data_cols}")

    df = load_data(
        test_file_path,
        data_cols,
        None,
        split_col,
        split_val,
        data_encoders,
        id_col=id_col,
    )

    x = df[data_cols].copy()

    model = model_dict["model"]

    for col in data_cols:
        x[col] = data_encoders[col].transform(x[col])

    label_encoder = model_dict["le"]

    p = model.predict(x)
    p_labels = label_encoder.inverse_transform(p)

    df["predicted"] = p_labels

    output_name = f"{split_val}.predicted.tsv"
    result_file_path = os.path.join(output_dir, output_name)
    df.to_csv(result_file_path, sep="\t", index=False)

    return p_labels


def main():
    """Read arguments, perform action."""
    model_dict = None

    # cmd
    args = parse_args()
    output_dir = args.output_dir
    output_dir = os.path.join(output_dir, args.target)
    os.makedirs(output_dir, exist_ok=True)

    # train
    if args.train is not None:
        model_dict = train_model(
            output_dir=output_dir,
            train_file_path=args.data,
            data_cols=args.cols,
            target_col=args.target,
            split_col=args.col_split,
            split_val=args.train,
            tune_val=args.tune,
            n_jobs=args.n_jobs,
        )

    # load model
    if args.model_load is not None:
        print(f"Loading model from: {args.model_load}")
        model_dict = pickle.load(open(args.model_load, "rb"))

    # eval
    if args.eval is not None:
        if model_dict is None:
            print("No model to eval")
            return

        results = eval_model(
            output_dir=output_dir,
            model_dict=model_dict,
            test_file_path=args.data,
            data_cols=args.cols,
            target_col=args.target,
            split_col=args.col_split,
            split_val=args.eval,
        )

    if args.classify is not None:
        if model_dict is None:
            print("No model to classify")
            return

        classify_csv(
            output_dir=output_dir,
            model_dict=model_dict,
            test_file_path=args.data,
            data_cols=args.cols,
            split_col=args.col_split,
            split_val=args.classify,
            id_col=args.col_id,
        )


if __name__ == "__main__":
    main()
