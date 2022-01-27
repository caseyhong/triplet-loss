# triplet-loss

Based on https://github.com/UKPLab/sentence-transformers. The main modification is for *relative* triplet loss - instead of picking triplets (a,p,n) based on discrete classes, choose triplets based on how close or far the target value of the candidate is to the target value of the anchor. This parameter, `target_margin`, is tuneable, but for now I have elected to go with 1 standard deviation for the entire dataset.
