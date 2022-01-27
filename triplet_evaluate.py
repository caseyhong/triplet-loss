from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)
import os
import csv
from enum import Enum
import logging
from typing import Union, Tuple, List, Iterable, Dict

from triplet_data import *

logger = logging.getLogger(__name__)


class SimilarityFunction(Enum):
    COSINE = 0
    EUCLIDEAN = 1
    MANHATTAN = 2
    DOT_PRODUCT = 3


class SentenceEvaluator:
    """
    Base class for all evaluators
    Extend this class and implement __call__ for custom evaluators.
    """

    def __call__(
        self, model, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> float:
        """
        This is called during training to evaluate the model.
        It returns a score for the evaluation with a higher score indicating a better result.
        :param model:
            the model to evaluate
        :param output_path:
            path where predictions and metrics are written to
        :param epoch
            the epoch where the evaluation takes place.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation on test data.
        :param steps
            the steps in the current epoch at time of the evaluation.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation at the end of the epoch.
        :return: a score for the evaluation with a higher score indicating a better result
        """
        pass


class TripletEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on a triplet: (a, p, n).
    Checks if distance(a, p) < distance(a, n)
    """

    def __init__(
        self,
        anchors: List[str],
        positives: List[str],
        negatives: List[str],
        main_distance_function: SimilarityFunction = None,
        name: str = "",
        batch_size: int = 16,
        show_progress_bar: bool = False,
        write_csv: bool = True,
    ):
        """
        Args:
            anchors: Sentences to check similarity to
            positives: List of positive sentences
            negatives: List of negative sentences
            main_distance_function: One of 0 (Cosine), 1 (Euclidean) or 2 (Manhattan). Defaults to None, returning all 3.
        """
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives
        self.name = name

        assert len(self.anchors) == len(self.positives)
        assert len(self.anchors) == len(self.negatives)

        self.main_distance_function = main_distance_function

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file: str = (
            "triplet_evaluation" + ("_" + name if name else "") + "_results.csv"
        )
        self.csv_headers = [
            "epoch",
            "steps",
            "accuracy_cosinus",
            "accuracy_manhattan",
            "accuracy_euclidean",
        ]
        self.write_csv = write_csv

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        anchors = []
        positives = []
        negatives = []

        for example in examples:
            anchors.append(example.texts[0])
            positives.append(example.texts[1])
            negatives.append(example.texts[2])
        return cls(anchors, positives, negatives, **kwargs)

    def __call__(
        self, model, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info(
            "TripletEvaluator: Evaluating the model on "
            + self.name
            + " dataset"
            + out_txt
        )

        num_triplets = 0
        (
            num_correct_cos_triplets,
            num_correct_manhatten_triplets,
            num_correct_euclidean_triplets,
        ) = (0, 0, 0)

        embeddings_anchors = model.encode(
            self.anchors,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )
        embeddings_positives = model.encode(
            self.positives,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )
        embeddings_negatives = model.encode(
            self.negatives,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )

        # Cosine distance
        pos_cos_distance = paired_cosine_distances(
            embeddings_anchors, embeddings_positives
        )
        neg_cos_distances = paired_cosine_distances(
            embeddings_anchors, embeddings_negatives
        )

        # Manhattan
        pos_manhattan_distance = paired_manhattan_distances(
            embeddings_anchors, embeddings_positives
        )
        neg_manhattan_distances = paired_manhattan_distances(
            embeddings_anchors, embeddings_negatives
        )

        # Euclidean
        pos_euclidean_distance = paired_euclidean_distances(
            embeddings_anchors, embeddings_positives
        )
        neg_euclidean_distances = paired_euclidean_distances(
            embeddings_anchors, embeddings_negatives
        )

        for idx in range(len(pos_cos_distance)):
            num_triplets += 1

            if pos_cos_distance[idx] < neg_cos_distances[idx]:
                num_correct_cos_triplets += 1

            if pos_manhattan_distance[idx] < neg_manhattan_distances[idx]:
                num_correct_manhatten_triplets += 1

            if pos_euclidean_distance[idx] < neg_euclidean_distances[idx]:
                num_correct_euclidean_triplets += 1

        accuracy_cos = num_correct_cos_triplets / num_triplets
        accuracy_manhattan = num_correct_manhatten_triplets / num_triplets
        accuracy_euclidean = num_correct_euclidean_triplets / num_triplets

        logger.info("Accuracy Cosine Distance:   \t{:.2f}".format(accuracy_cos * 100))
        logger.info(
            "Accuracy Manhattan Distance:\t{:.2f}".format(accuracy_manhattan * 100)
        )
        logger.info(
            "Accuracy Euclidean Distance:\t{:.2f}\n".format(accuracy_euclidean * 100)
        )

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow(
                        [
                            epoch,
                            steps,
                            accuracy_cos,
                            accuracy_manhattan,
                            accuracy_euclidean,
                        ]
                    )

            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            epoch,
                            steps,
                            accuracy_cos,
                            accuracy_manhattan,
                            accuracy_euclidean,
                        ]
                    )

        if self.main_distance_function == SimilarityFunction.COSINE:
            return accuracy_cos
        if self.main_distance_function == SimilarityFunction.MANHATTAN:
            return accuracy_manhattan
        if self.main_distance_function == SimilarityFunction.EUCLIDEAN:
            return accuracy_euclidean

        return max(accuracy_cos, accuracy_manhattan, accuracy_euclidean)