import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_metric_learning import losses

from math import log

# class SelfSupervisedContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super(SelfSupervisedContrastiveLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, feature_vectors, labels):
#         # Normalize feature vectors
#         feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
#         # Compute logits
#         logits = torch.div(
#             torch.matmul(
#                 feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
#             ),
#             self.temperature,
#         )
#         return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(labels))

class SelfSupervisedContrastiveLoss(nn.Module):
    """
    Modified implementation of the following:
        https://github.com/HobbitLong/SupContrast
    """
    def __init__(self,
                 temperature: float = 0.07,
                 distributed: bool = False,
                 local_rank: int = 0):
        super(SelfSupervisedContrastiveLoss, self).__init__()

        self.temperature = temperature
        self.distributed = distributed
        self.local_rank = local_rank

    def forward(self, features: torch.FloatTensor):
        _, num_views, _ = features.size()

        # Normalize features to lie on a unit hypersphere.
        features = F.normalize(features, dim=-1)
        features = torch.cat(torch.unbind(features, dim=1), dim=0)       # (B, N, F) -> (NB, F)
        contrasts = features

        # Compute logits (aka. similarity scores) & numerically stabilize them
        logits = features @ contrasts.T  # (BN, F) x (F, NB * world_size)
        logits = logits.div(self.temperature)

        # Compute masks
        _, pos_mask, neg_mask = self.create_masks(logits.size(), self.local_rank, num_views)

        # Compute loss
        numerator = logits * pos_mask  # FIXME
        denominator = torch.exp(logits) * pos_mask.logical_or(neg_mask)
        denominator = denominator.sum(dim=1, keepdim=True)
        log_prob = numerator - torch.log(denominator)
        mean_log_prob = (log_prob * pos_mask) / pos_mask.sum(dim=1, keepdim=True)
        loss = torch.neg(mean_log_prob)
        loss = loss.sum(dim=1).mean()

        return loss

    @staticmethod
    @torch.no_grad()
    def create_masks(shape, local_rank: int, num_views: int = 2):

        device = local_rank
        nL, nG = shape

        local_mask = torch.eye(nL // num_views, device=device).repeat(2, 2)  # self+positive indicator
        local_pos_mask = local_mask - torch.eye(nL, device=device)           # positive indicator
        local_neg_mask = torch.ones_like(local_mask) - local_mask            # negative indicator

        # Global mask of self+positive indicators
        global_mask = torch.zeros(nL, nG, device=device)
        global_mask[:, nL*local_rank:nL*(local_rank+1)] = local_mask

        # Global mask of positive indicators
        global_pos_mask = torch.zeros_like(global_mask)
        global_pos_mask[:, nL*local_rank:nL*(local_rank+1)] = local_pos_mask

        # Global mask of negative indicators
        global_neg_mask = torch.ones_like(global_mask)
        global_neg_mask[:, nL*local_rank:nL*(local_rank+1)] = local_neg_mask

        return global_mask, global_pos_mask, global_neg_mask


    @staticmethod
    def semisupervised_mask(unlabeled_size: int, labels: torch.Tensor):
        """Create mask for semi-supervised contrastive learning."""

        labels = labels.view(-1, 1)
        labeled_size = labels.size(0)
        mask_size = unlabeled_size + labeled_size
        mask = torch.zeros(mask_size, mask_size, dtype=torch.float32).to(labels.device)

        L = torch.eq(labels, labels.T).float()
        mask[unlabeled_size:, unlabeled_size:] = L
        U = torch.eye(unlabeled_size, dtype=torch.float32).to(labels.device)
        mask[:unlabeled_size, :unlabeled_size] = U
        mask.clamp_(0, 1)  # Just in case. This might not be necessary.

        return mask