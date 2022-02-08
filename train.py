import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import time
from datetime import datetime
import logging
import os
import os.path as osp
from sentence_transformers import SentenceTransformer, LoggingHandler
import torch
from torch.utils.data import DataLoader
import wandb
import gc

from triplet_data import *
from triplet_evaluate import *
from triplet_loss import *

logger = logging.getLogger(__name__)


def get_examples(X, y):
    """Take X (pd.Series) and y (pd.Series) and return a list of InputExamples"""
    ret = []
    for i, data in enumerate(zip(X, y)):
        ret.append(InputExample(guid=i, texts=[data[0]], label=data[1]))
    return ret


def get_input_examples(
    df,
    text_col="clean_text",
    target_col="num_replies",
    # target_margin=1,
    val_prop=0.2,
    test_prop=0.2,
    random_state=42,
):
    logger.info("Loading dataset")
    start = time.time()
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df[target_col],
        test_size=test_prop,
        random_state=random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_prop, random_state=random_state
    )

    start = time.time()
    train_set = get_examples(X_train, y_train)
    dev_set = get_examples(X_val, y_val)
    test_set = get_examples(X_test, y_test)
    logger.info(
        f"Format train/val/test examples: {round(time.time()-start, 2)}s elapsed."
    )

    # cleanup
    del X_train
    del X_val
    del X_test
    del y_train
    del y_val
    del y_test
    gc.collect()

    start = time.time()
    # For dev & test set, we return triplets (anchor, positive, negative)
    random.seed(random_state)  # Fix seed, so that we always get the same triplets
    dev_triplets = triplets_from_labeled_dataset(dev_set)
    test_triplets = triplets_from_labeled_dataset(test_set)
    logger.info(
        f"Get triplets for dev/test set: {round(time.time()-start, 2)}s elapsed."
    )
    del dev_set
    del test_set
    gc.collect()

    return train_set, dev_triplets, test_triplets


def target2class(targets, log_norm=False):
    """
    Take targets (pd.Series) and transform into binary classification labels.
    Label is 0 if the target is within 1 std of the mean, 1 otherwise
    """
    logger.info("Transforming data targets to class labels")
    if log_norm:
        targets = np.log(targets + 1)
    return targets.apply(lambda x: abs(x - targets.mean()) <= targets.std()).astype(
        "int16"
    )


def triplets_from_labeled_dataset(input_examples):
    """Take input_examples (list[InputExample]) and form triplets for dev and test sets"""
    triplets = []
    label2sentence = defaultdict(list)
    for ie in input_examples:
        label2sentence[ie.label].append(ie)
    for ie in input_examples:
        anchor = ie
        if (
            len(label2sentence[ie.label]) < 2
        ):  # We need at least 2 examples per label to create a triplet
            continue
        positive = None
        while positive is None or positive.guid == anchor.guid:
            positive = random.choice(label2sentence[ie.label])

        negative = None
        while negative is None or negative.label == anchor.label:
            negative = random.choice(input_examples)

        triplets.append(
            InputExample(texts=[anchor.texts[0], positive.texts[0], negative.texts[0]])
        )
    return triplets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--target_var", type=str, nargs="?", default="num_replies")
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    parser.set_defaults(debug=False)

    # Parse
    args = parser.parse_args()

    # Params
    MODEL_NAME = "all-distilroberta-v1"
    BATCH_SIZE = 32
    NUM_EPOCHS = args.epochs
    TARGET_VAR = args.target_var  # either `num_replies` or `log_num_replies`
    LOSS = "BatchHardTripletLoss"
    TRAIN_FILE = "train/seq2reply_regression_data.pickle"
    DEBUG = args.debug

    pdate = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    project_name = f"finetune-{LOSS}-ufo-{MODEL_NAME}-{pdate}"
    output_path = osp.join("/output", project_name)
    os.makedirs(output_path, exist_ok=True)

    # setup logging to stdout
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    # load data
    data = pd.read_pickle(TRAIN_FILE)
    data = data.loc[data.clean_text.notnull()]
    data = data.loc[~(data.clean_text == "")]
    if DEBUG:
        data = data.sample(1000)
    data["label"] = target2class(data["num_replies"])
    logging.info(
        f"Loaded {data.shape[0]} rows from file. {data.loc[data.label==0].shape[0]} with label 0. {data.loc[data.label==1].shape[0]} with label 1."
    )

    train_set, dev_set, test_set = get_input_examples(
        data, text_col="clean_text", target_col="label"
    )

    # cleanup
    del data
    gc.collect()

    # Initialize a special wrapper "SentenceLabelDataset" for train_set
    # It will yield batches that contain at least two samples with the same label
    train_data_sampler = SentenceLabelDataset(train_set)

    # Initialize dataloader
    train_dataloader = DataLoader(
        train_data_sampler, batch_size=BATCH_SIZE, drop_last=True
    )

    # Initialize model
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    # Initialize wandb
    wandb.init(project=project_name, entity="jisoo")
    wandb.watch(model)

    # Initialize loss
    train_loss = BatchHardTripletLoss(model=model, wandb=wandb)

    logging.info("Read val dataset")
    dev_evaluator = TripletEvaluator.from_input_examples(
        dev_set, name="ufo-dev", main_distance_function=1
    )

    logging.info("Performance before fine-tuning:")
    dev_evaluator(model)

    warmup_steps = int(len(train_dataloader) * 0.1)  # 10% of train data
    logging.info(f"Warmup steps/total: {warmup_steps}/{len(train_dataloader)}")

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        epochs=NUM_EPOCHS,
        evaluation_steps=1000,
        warmup_steps=warmup_steps,
        output_path=output_path,
        checkpoint_path=output_path,
        checkpoint_save_steps=1000,
        checkpoint_save_total_limit=3,
    )
