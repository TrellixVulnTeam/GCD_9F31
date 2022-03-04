import numpy as np
# from sskmeans import KMeans
import argparse
import os
import torch as th

from data.cifarloader import CIFAR10SampledSetLoader, CIFAR10SampledSetLoaderNoAugmentation, CIFAR100Loader
from sklearn.metrics import adjusted_mutual_info_score
import vision_transformer as vits
from tqdm import tqdm
from vision_transformer import DINOHead
import torch.nn as nn
import numpy as np
import vision_transformer as vits
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

def get_reference_dict(clusters,data_label):
    reference_label = {}
    # For loop to run through each label of cluster label
    for i in range(len(np.unique(clusters))):
        index = np.where(clusters == i,1,0)
        num = np.bincount(data_label[index==1]).argmax()
        reference_label[i] = num
    return reference_label

# Mapping predictions to original labels
def get_labels(clusters, reference_labels):
    temp_labels = np.random.rand(len(clusters))
    for i in range(len(clusters)):
        temp_labels[i] = reference_labels[clusters[i]]
    return temp_labels

def cluster_acc(y_true, y_pred):
    """
    This is code from original repository.
    Calculate clustering accuracy. Require scipy installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)

    accuracy = 0
    for idx in range(len(ind[0]) - 1):
        i = ind[0][idx]
        j = ind[1][idx]
        accuracy += w[i, j]
    accuracy = accuracy * 1.0 / y_pred.size
    return accuracy

def get_model(pretrained=True, num_classes=10, **kwargs):
    model = vits.__dict__["vit_base"](patch_size=16, num_classes=0, **kwargs)
    if pretrained:
        state_dict = th.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    #head will get discarded
    model.head = DINOHead(in_dim=model.num_features, out_dim=num_classes)
    return model

def get_features(model, args):  
    model.eval()
    model.to(args.device)
    print("Start extracting Feature")
    labeled_loader, unlabeled_loader, all_loader = CIFAR10SampledSetLoaderNoAugmentation(root=args.dataset_root, 
                                                                                        batch_size=args.batch_size, 
                                                                                        aug=None, 
                                                                                        num_workers=args.num_workers,
                                                                                        shuffle=False)
    

    labels = []
    features = []
    for l_batch_idx, (image, target, idx) in enumerate(tqdm(labeled_loader)):
        image = image.to(args.device)
        with th.no_grad():
            feats = model(image)
        feats = feats.detach()
        features.append(feats)
        # feature_vector.extend(feats.cpu().detach().numpy())
        labels.extend(target.numpy())
        if l_batch_idx % 20 == 0:
            print(f"Labeled Step [{l_batch_idx}/{len(labeled_loader)}]\t Computing features...")

    features = th.cat(features)
    features = nn.functional.normalize(features, dim=1, p=2)
    features = features.cpu().detach().numpy()
    
    labels = np.array(labels)
    return features, labels

def load_pretrained_weights(model, args):
    checkpoint = args.pretrained_weights + '{dataset}_last.pth'.format(dataset=args.dataset)
    assert os.path.isfile(checkpoint), "Checkpoint does not exist"
    if os.path.isfile(checkpoint):
        state_dict = th.load(checkpoint, map_location="cpu")
        state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=True)
    else:
        print("There is no reference weights available for this model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=["cifar10", "cifar100"],
        help="dataset name",
    )
    parser.add_argument('--pretrained_weights', default='./checkpoint/', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--lr", default=1e-1, type=float)
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for SGD")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum for SGD")
    parser.add_argument("--batch_size", default=128, type=int, help="On the contrastive step this will be multiplied by two.")
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers for Dataloader")
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
   
    args = parser.parse_args()

    device = "cuda" if th.cuda.is_available() else "cpu"
    args.device = device

    model = get_model()
    model.cuda()
    load_pretrained_weights(model, args)
    # model.head = nn.Identity()
    # for param in model.parameters():
    #     print(param)
    # model.eval()
    features, labels = get_features(model, args)
    for i in range(2, 51):
        kmeans = KMeans(n_clusters=i)
        clusters = kmeans.fit_predict(features)
        reference_labels = get_reference_dict(clusters, labels)
        predicted_labels = get_labels(clusters, reference_labels)
        print("k {} acc {}".format(i, accuracy_score(predicted_labels,labels)))
    