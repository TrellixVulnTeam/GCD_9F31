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
    # mat = confusion_matrix(labels,predicted_labels)
    # print(mat)
    # print(kmeans.cluster_centers_.shape)

    # x_data = [i for i in range(768)]
    # plt.scatter(x_data,kmeans.cluster_centers_[0], color = 'red',alpha=0.2,s=70)
    # plt.scatter(x_data,kmeans.cluster_centers_[1] , color = 'blue',alpha=0.2,s=50)
    # plt.scatter(x_data,kmeans.cluster_centers_[2], color = 'green',alpha=0.2,s=70)
    # plt.scatter(x_data,kmeans.cluster_centers_[3] , color = 'yellow',alpha=0.2,s=50)
    # plt.scatter(x_data,kmeans.cluster_centers_[4], color = 'black',alpha=0.2,s=70)
    # plt.scatter(x_data,kmeans.cluster_centers_[5] , color = 'brown',alpha=0.2,s=50)
    # plt.scatter(x_data,kmeans.cluster_centers_[6], color = 'cyan',alpha=0.2,s=70)
    # plt.scatter(x_data,kmeans.cluster_centers_[7] , color = 'magenta',alpha=0.2,s=50)
    # plt.scatter(x_data,kmeans.cluster_centers_[8], color = '#9467bd',alpha=0.2,s=70)
    # plt.scatter(x_data,kmeans.cluster_centers_[9] , color = '#8c564b',alpha=0.2,s=50)
    # plt.savefig("clusters.png")

    # sse = []
    # list_k = [1,2,3,4,5,6,7,8,9,10]
    # for k in list_k:
    #     km = KMeans(n_clusters=k)
    #     clusters = km.fit_predict(reshaped_data)
    #     sse.append(km.inertia_)
    
    # reference_labels = get_reference_dict(clusters,data_label)
    # predicted_labels = get_labels(clusters,reference_labels)
    
    # print(f"Accuracy for k = {k}: ", accuracy_score(predicted_labels,data_label))

    # # Plot sse against k
    # plt.figure(figsize=(6, 6))
    # plt.plot(list_k, sse, '-o')
    # plt.xlabel(r'Number of clusters *k*')
    # plt.ylabel('Sum of squared distance');

    # for i in range(1, 100):
    #     pred = KMeans(i).fit_predict(features)
    #     accuracy = cluster_acc(labels, pred)
    #     print("Accuracy k={}: {}".format(i, accuracy))
    # print(len(features))
    # print(len(labels))
    # kmeans = KMeans(k=9).fit(features)
    # centroids = kmeans.k
    # print(len(centroids))

    # # # plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
    # # # plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
    # # # plt.show()

    # for i in range(1, 10):
    #     kmeans = KMeans(n_clusters=i)
    #     kmeans_results = kmeans.fit_predict(features)
    #     print ("\nK={} Score:\t{}\n\n".format(i, adjusted_mutual_info_score(labels, kmeans_results)))

    # # Fit K-means with Scikit
    # X = features
    # kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
    # result = kmeans.fit_predict(features)
    # centroids = kmeans.cluster_centers_
    # print(len(centroids))
    # print ("\nK={} Score:\t{}\n\n".format(10, adjusted_mutual_info_score(labels, result)))

    # # Predict the cluster for all the samples
    # P = kmeans.predict(X)

    # Generate scatter plot for training data
    # colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', P))
    # plt.scatter(X[:,0], X[:,1], c=colors, marker="o", picker=True)
    # plt.title('Two clusters of data')
    # plt.xlabel('Temperature yesterday')
    # plt.ylabel('Temperature today')
    # plt.show()
    # plt.savefig("plot.png")

    # labeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(5))
    # model.eval()
    # preds=np.array([])
    # targets=np.array([])
    # for batch_idx, (x, label, _) in enumerate(tqdm(labeled_eval_loader)):
    #     x, label = x.to(device), label.to(device)
    #     output = model.head.forward(model(x))
    #     _, pred = output.max(1)
    #     targets=np.append(targets, label.cpu().numpy())
    #     preds=np.append(preds, pred.cpu().numpy())
    # acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds) 
    # print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))