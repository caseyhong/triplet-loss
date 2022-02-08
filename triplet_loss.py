import logging
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict

logger = logging.getLogger(__name__)


class BatchHardTripletLossDistanceFunction:
    """
    This class defines distance functions, that can be used with Batch[All/Hard/SemiHard]TripletLoss
    """

    @staticmethod
    def cosine_distance(embeddings):
        """
        Compute the 2D matrix of cosine distances (1-cosine_similarity) between all embeddings.
        """
        return 1 - util.pytorch_cos_sim(embeddings, embeddings)

    @staticmethod
    def euclidian_distance(embeddings, squared=False):
        """
        Compute the 2D matrix of euclidian distances between all the embeddings.
        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """

        dot_product = torch.matmul(embeddings, embeddings.t())

        # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        # shape (batch_size,)
        square_norm = torch.diag(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = (
            square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        )

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances[distances < 0] = 0

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = distances.eq(0).float()
            distances = distances + mask * 1e-16

            distances = (1.0 - mask) * torch.sqrt(distances)

        return distances


class BatchHardTripletLoss(nn.Module):
    """
    BatchHardTripletLoss takes a batch with (label, sentence) pairs and computes the loss for all possible,
    valid triplets, i.e. anchor and positive must be within `target_margin` of each other, anchor and negative
    must be further apart than 2*`target_margin` of each other.

    Source: https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
    Paper: In Defense of the Triplet Loss for Person Re-Identification, https://arxiv.org/abs/1703.07737
    Blog post: https://omoindrot.github.io/triplet-loss
    """

    def __init__(
        self,
        model: SentenceTransformer,
        # target_margin: float,
        wandb,
        distance_metric=BatchHardTripletLossDistanceFunction.euclidian_distance,
        margin: float = 5,
    ):
        super(BatchHardTripletLoss, self).__init__()
        self.sentence_embedder = model
        # self.target_margin = target_margin
        self.triplet_margin = margin
        self.distance_metric = distance_metric
        self.wandb = wandb

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        rep = self.sentence_embedder(sentence_features[0])["sentence_embedding"]
        triplet_loss = self.batch_hard_triplet_loss(labels, rep)
        self.wandb.log({"triplet_loss": triplet_loss})
        return triplet_loss

    # Hard Triplet Loss
    # Source: https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
    # Paper: In Defense of the Triplet Loss for Person Re-Identification, https://arxiv.org/abs/1703.07737
    # Blog post: https://omoindrot.github.io/triplet-loss
    def batch_hard_triplet_loss(self, labels: Tensor, embeddings: Tensor) -> Tensor:
        """Build the triplet loss over a batch of embeddings.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                If false, output is the pairwise euclidiean distance matrix.
        Returns:
            Label_Sentence_Triplet: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = self.distance_metric(embeddings)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive
        # mask_anchor_positive = BatchHardTripletLoss.get_anchor_positive_triplet_mask(
        #     labels, self.target_margin
        # ).float()
        mask_anchor_positive = BatchHardTripletLoss.get_anchor_positive_triplet_mask(
            labels
        ).float()

        # We put to 0 any element where (a, p) is not valid
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative
        # mask_anchor_negative = BatchHardTripletLoss.get_anchor_negative_triplet_mask(
        #     labels, self.target_margin
        # ).float()
        mask_anchor_negative = BatchHardTripletLoss.get_anchor_negative_triplet_mask(
            labels
        ).float()

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
            1.0 - mask_anchor_negative
        )

        # shape (batch_size,)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        tl = hardest_positive_dist - hardest_negative_dist + self.triplet_margin
        tl[tl < 0] = 0
        triplet_loss = tl.mean()

        return triplet_loss

    # @staticmethod
    # def get_anchor_positive_triplet_mask(labels, target_margin):
    #     """
    #     Return a 2D mask where mask[a,p] is True iff a and p are distinct and are within `target_margin`
    #     of each other
    #     Args:
    #         labels: tf.int32 `Tensor` with shape [batch_size]
    #         target_margin: target margin
    #     Returns:
    #         mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    #     """
    #     indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    #     indices_not_equal = ~indices_equal

    #     # Check if abs(labels[i] - labels[j]) <= target_margin
    #     # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    #     diff = (torch.abs(labels - labels.unsqueeze(1)) - target_margin) <= 0
    #     valid_labels = diff.unsqueeze(0) & diff.unsqueeze(1)

    #     return indices_not_equal & valid_labels

    @staticmethod
    def get_anchor_positive_triplet_mask(labels):
        """
        Return a 2D mask where mask[a,p] is True iff a and p are distinct and have the same label.`
        of each other
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check that i and j are distinct
        indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
        indices_not_equal = ~indices_equal

        # Check if labels[i] == labels[j]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

        return labels_equal & indices_not_equal

    # @staticmethod
    # def get_anchor_negative_triplet_mask(labels, target_margin):
    #     # Check if abs(labels[i] - labels[j]) >= 2*target_margin
    #     diff = (torch.abs(labels - labels.unsqueeze(1)) - 2 * target_margin) >= 0
    #     valid_labels = diff.unsqueeze(0) & diff.unsqueeze(1)
    #     return ~valid_labels

    @staticmethod
    def get_anchor_negative_triplet_mask(labels):
        """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check if labels[i] != labels[k]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)

        return ~(labels.unsqueeze(0) == labels.unsqueeze(1))


class BatchAllTripletLoss(nn.Module):
    """
    BatchAllTripletLoss takes a batch with (label, sentence) pairs and computes the loss for all possible,
    valid triplets, i.e. anchor and positive must be within `target_margin` of each other, anchor and negative
    must be further apart than `target_margin` of each other.

    Source: https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
    Paper: In Defense of the Triplet Loss for Person Re-Identification, https://arxiv.org/abs/1703.07737
    Blog post: https://omoindrot.github.io/triplet-loss
    """

    def __init__(
        self,
        model: SentenceTransformer,
        target_margin: float,
        wandb,
        distance_metric=BatchHardTripletLossDistanceFunction.euclidian_distance,
        margin: float = 5,
    ):
        super(BatchAllTripletLoss, self).__init__()
        self.sentence_embedder = model
        self.target_margin = target_margin
        self.triplet_margin = margin
        self.distance_metric = distance_metric
        self.wandb = wandb

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        rep = self.sentence_embedder(sentence_features[0])["sentence_embedding"]
        triplet_loss = self.batch_all_triplet_loss(labels, rep)
        self.wandb.log({"triplet_loss": triplet_loss})
        return triplet_loss

    @staticmethod
    def get_triplet_mask(labels, target_margin):
        """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
        A triplet (i, j, k) is valid if:
            - i, j, k are distinct
            - |labels[i]-labels[j]| < target_margin and |labels[i] - labels[k]| > target_margin
            - TODO: implement asymmetric

        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
            target_margin: target margin
        """
        # Check that i, j, and k are distinct
        indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
        indices_not_equal = ~indices_equal
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)

        distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

        # Now mask according to labels
        diff = (torch.abs(labels - labels.unsqueeze(1)) - target_margin) <= 0
        valid_labels = (diff.unsqueeze(2) & diff.unsqueeze(1)) & diff.unsqueeze(0)

        return distinct_indices & valid_labels

    def batch_all_triplet_loss(self, labels, embeddings):
        """Build the triplet loss over a batch of embeddings.
        We generate all valid triplets and average the loss over the positive ones.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
        Returns:
            Label_Sentence_Triplet: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = self.distance_metric(embeddings)

        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.triplet_margin

        # Put to zero the invalid triplets
        # (where label(a) and label(p) are NOT within `target_margin` of each other; or
        # label(a) and label(n) are within `target_margin` of each other, or a == p)
        mask = BatchHardTripletLoss.get_triplet_mask(labels, self.target_margin)
        triplet_loss = mask.float() * triplet_loss

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss[triplet_loss < 0] = 0

        # Count the number of positive triplets (where triplet_loss > 0)
        valid_triplets = triplet_loss[triplet_loss > 1e-16]
        num_positive_triplets = valid_triplets.size(0)
        num_valid_triplets = mask.sum()

        fraction_positive_triplets = num_positive_triplets / (
            num_valid_triplets.float() + 1e-16
        )

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

        return triplet_loss
