import numpy as np
import logging
from collections import defaultdict
from typing import Union, Tuple, List, Iterable, Dict
from torch.utils.data import IterableDataset
from tqdm import tqdm
import itertools
import time

logger = logging.getLogger(__name__)


class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """

    def __init__(
        self, guid: str = "", texts: List[str] = None, label: Union[int, float] = 0
    ):
        """
        Creates one InputExample with the given texts, guid and label
        :param guid
            id for the example
        :param texts
            the texts for the example. Note, str.strip() is called on the texts
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = texts
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(
            str(self.label), "; ".join(self.texts)
        )


class SentenceLabelDataset(IterableDataset):
    """
    This dataset can be used for some specific Triplet Losses like BATCH_HARD_TRIPLET_LOSS which requires
    multiple examples with the same label in a batch.
    It draws n consecutive, random and unique samples from one label at a time. This is repeated for each label.
    Labels with fewer than n unique samples are ignored.
    This also applied to drawing without replacement, once less than n samples remain for a label, it is skipped.
    This *DOES NOT* check if there are more labels than the batch is large or if the batch size is divisible
    by the samples drawn per label.
    """

    def __init__(
        self,
        examples: List[InputExample],
        target_margin: float,
        samples_per_label: int = 2,
        with_replacement: bool = False,
    ):
        """
        Creates a LabelSampler for a SentenceLabelDataset.
        :param examples:
            a list with InputExamples
        :param samples_per_label:
            the number of consecutive, random and unique samples drawn per label. Batch size should be a multiple of samples_per_label
        :param with_replacement:
            if this is True, then each sample is drawn at most once (depending on the total number of samples per label).
            if this is False, then one sample can be drawn in multiple draws, but still not multiple times in the same
            drawing.
        """
        super().__init__()

        self.samples_per_label = samples_per_label
        self.target_margin = target_margin

        start = time.time()

        # Group examples by label
        label2ex = defaultdict(list)
        lp = sum(1 for _ in itertools.permutations(examples, 2))
        for i, j in tqdm(
            itertools.permutations(examples, 2),
            total=lp,
            desc="SentenceLabelDataset [label2ex]",
        ):
            if abs(i.label - j.label) <= self.target_margin:
                label2ex[i.label].append(j)

        # for example in examples:
        #     for tmp_example in examples:
        #         if tmp_example.guid == example.guid:
        #             continue
        #         if abs(tmp_example.label - example.label) <= self.target_margin:
        #             label2ex[example.label].append(tmp_example)

        # Include only labels with at least 2 examples
        self.grouped_inputs = []
        self.groups_right_border = []
        num_labels = 0

        for label, label_examples in label2ex.items():
            if len(label_examples) >= self.samples_per_label:
                self.grouped_inputs.extend(label_examples)
                self.groups_right_border.append(
                    len(self.grouped_inputs)
                )  # At which position does this label group / bucket end?
                num_labels += 1

        self.label_range = np.arange(num_labels)
        self.with_replacement = with_replacement
        np.random.shuffle(self.label_range)

        logger.info(
            f"Initialized SentenceLabelDataset: {round(time.time()-start,2)}s elapsed."
        )

        logger.info(
            "SentenceLabelDataset: {} examples, from which {} examples could be used (those labels appeared at least {} times). {} different labels found.".format(
                len(examples),
                len(self.grouped_inputs),
                self.samples_per_label,
                num_labels,
            )
        )

    def __iter__(self):
        label_idx = 0
        count = 0
        already_seen = {}
        while count < len(self.grouped_inputs):
            label = self.label_range[label_idx]
            if label not in already_seen:
                already_seen[label] = set()

            left_border = 0 if label == 0 else self.groups_right_border[label - 1]
            right_border = self.groups_right_border[label]

            if self.with_replacement:
                selection = np.arange(left_border, right_border)
            else:
                selection = [
                    i
                    for i in np.arange(left_border, right_border)
                    if i not in already_seen[label]
                ]

            if len(selection) >= self.samples_per_label:
                for element_idx in np.random.choice(
                    selection, self.samples_per_label, replace=False
                ):
                    count += 1
                    already_seen[label].add(element_idx)
                    yield self.grouped_inputs[element_idx]

            label_idx += 1
            if label_idx >= len(self.label_range):
                label_idx = 0
                already_seen = {}
                np.random.shuffle(self.label_range)

    def __len__(self):
        return len(self.grouped_inputs)
