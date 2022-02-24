import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_metric_learning import losses

from math import log

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        # Compute logits
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.temperature,
        )
        return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(labels))


# class SupervisedContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         """
#         Implementation of the loss described in the paper Supervised Contrastive Learning :
#         https://arxiv.org/abs/2004.11362
#         :param temperature: int
#         """
#         super(SupervisedContrastiveLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, projections, targets):
#         """
#         :param projections: torch.Tensor, shape [batch_size, projection_dim]
#         :param targets: torch.Tensor, shape [batch_size]
#         :return: torch.Tensor, scalar
#         """
        
#         device = torch.device("cuda:2") if projections.is_cuda else torch.device("cpu")

#         dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
#         # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
#         exp_dot_tempered = (
#             torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
#         )

#         mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
#         mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
#         mask_combined = mask_similar_class * mask_anchor_out
#         cardinality_per_samples = torch.sum(mask_combined, dim=1)

#         log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
#         supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
#         supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

#         return supervised_contrastive_loss