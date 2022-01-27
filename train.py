import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
from datetime import datetime
import logging
import os
import os.path as osp
from sentence_transformers import SentenceTransformer, LoggingHandler
import torch
from torch.utils.data import DataLoader
import wandb

from triplet_data import *
from triplet_evaluate import *
from triplet_loss import *

logger = logging.getLogger(__name__)


def get_input_examples(
    df,
    text_col="clean_text",
    target_col="num_replies",
    target_margin=1,
    val_prop=0.1,
    test_prop=0.2,
    random_state=42,
):
    logger.info("Loading dataset")
    start = time.time()
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df[target_col].astype(float),
        test_size=test_prop,
        random_state=random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_prop, random_state=random_state
    )
    logger.info(f"Split train/val/test: {round(time.time()-start, 2)}s elapsed.")

    def get_examples(X, y):
        """Take X (pd.Series) and y (pd.Series) and return a list of InputExamples"""
        ret = []
        for i, data in enumerate(zip(X, y)):
            ret.append(InputExample(guid=i, texts=[data[0]], label=data[1]))
        return ret

    start = time.time()
    train_set = get_examples(X_train, y_train)
    dev_set = get_examples(X_val, y_val)
    test_set = get_examples(X_test, y_test)
    logger.info(
        f"Format train/val/test examples: {round(time.time()-start, 2)}s elapsed."
    )

    def triplets_from_labeled_dataset(input_examples, target_margin):
        triplets = []
        pos_dict = defaultdict(list)
        neg_dict = defaultdict(list)
        for i in input_examples:
            for j in input_examples:
                if j.guid == i.guid:
                    continue
                if abs(int(j.label) - int(i.label)) <= target_margin:
                    pos_dict[i.label].append(j)
                if abs(int(j.label) - int(i.label)) >= 2 * target_margin:
                    neg_dict[i.label].append(j)

        for anchor in input_examples:
            if (
                len(pos_dict[anchor.label]) < 2
            ):  # we need at least 2 examples per label to create a triplet
                continue

            positive = None
            while positive is None or positive.guid == anchor.guid:
                positive = random.choice(pos_dict[anchor.label])

            negative = None
            while negative is None or negative.guid == anchor.guid:
                negative = random.choice(neg_dict[anchor.label])

            triplets.append(
                InputExample(
                    texts=[anchor.texts[0], positive.texts[0], negative.texts[0]]
                )
            )
        return triplets

    start = time.time()
    # For dev & test set, we return triplets (anchor, positive, negative)
    random.seed(42)  # Fix seed, so that we always get the same triplets
    dev_triplets = triplets_from_labeled_dataset(dev_set, target_margin)
    test_triplets = triplets_from_labeled_dataset(test_set, target_margin)
    logger.info(
        f"Get triplets for dev/test set: {round(time.time()-start, 2)}s elapsed."
    )

    return train_set, dev_triplets, test_triplets


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
    output_path = osp.join("output", project_name)
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
    logging.info(f"Loaded {data.shape[0]} rows from {TRAIN_FILE}")
    data["log_num_replies"] = np.log(data.num_replies + 1)
    raw_std = data.num_replies.std()
    log_std = data.log_num_replies.std()
    target_margin_map = {
        "num_replies": raw_std,
        "log_num_replies": log_std,
    }  # map colname to target_margin
    TARGET_MARGIN = target_margin_map[TARGET_VAR]
    logging.info(f"Target margin: {TARGET_VAR} std = {round(TARGET_MARGIN, 3)}")

    train_set, dev_set, test_set = get_input_examples(
        data,
        target_margin=TARGET_MARGIN,
        text_col="clean_text",
        target_col="num_replies",
    )

    # We create a special dataset "SentenceLabelDataset" to wrap out train_set
    # It will yield batches that contain at least two samples with the same label
    train_data_sampler = SentenceLabelDataset(train_set, target_margin=TARGET_MARGIN)
    train_dataloader = DataLoader(
        train_data_sampler, batch_size=BATCH_SIZE, drop_last=True
    )

    model = SentenceTransformer(MODEL_NAME)

    ### Triplet losses ####################
    ### There are 4 triplet loss variants:
    ### - BatchHardTripletLoss
    ### - BatchHardSoftMarginTripletLoss - TODO
    ### - BatchSemiHardTripletLoss - TODO
    ### - BatchAllTripletLoss
    #######################################
    wandb.init(project=project_name, entity="jisoo")
    wandb.watch(model)

    LossDict = {
        "BatchHardTripletLoss": BatchHardTripletLoss(
            model=model, target_margin=TARGET_MARGIN, wandb=wandb
        ),
        "BatchAllTripletLoss": BatchAllTripletLoss(
            model=model, target_margin=TARGET_MARGIN, wandb=wandb
        ),
    }
    train_loss = LossDict[LOSS]

    logging.info("Read val dataset")
    dev_evaluator = TripletEvaluator.from_input_examples(
        dev_set, name="ufo-dev", main_distance_function=1
    )

    logging.info("Performance before fine-tuning:")
    dev_evaluator(model)

    warmup_steps = int(len(train_dataloader) * NUM_EPOCHS * 0.1)  # 10% of train data
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
