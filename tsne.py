import argparse
import time
import os
import sys
import torch

from data.cifarloader import CIFAR10Loader
import vision_transformer as vits
from tqdm import tqdm
from utils import get_model
from vision_transformer import DINOHead
from torch import optim
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_features(model, args):        # PyTorch在ImageNet上的pre-trained weight進行特徵萃取
    model.eval()
    model.to(args.device)

    features = None
    labels = []
    print("Start extracting Feature")
    s_labeled_loader = CIFAR10Loader(root=args.dataset_root, 
                                                batch_size=args.batch_size, 
                                                aug=None, 
                                                num_workers=args.num_workers, 
                                                shuffle=True)
    
    outputs = []
    labels = []
    with torch.no_grad():
        for batch_idx, (image, target, idx) in enumerate(tqdm(s_labeled_loader)):
            x, target = image.to(args.device), target.to(args.device)
            target = target.squeeze().tolist()
            for element in target:
                labels.append(element)
            output = model(x)
            current_features = output.cpu().numpy()
            if features is not None:
                features = np.concatenate((features, current_features))
            else:
                features = current_features

    return features, labels

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def visualize_tsne_points(tx, ty, labels):
    print('Plotting TSNE image')
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    class_name = ['airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships,trucks']

    colors_per_class = {
        0 : [254, 202, 87],
        1 : [255, 107, 107],
        2 : [10, 189, 227],
        3 : [255, 159, 243],
        4 : [16, 172, 132],
        5 : [128, 80, 128],
        6 : [254, 88, 87],
        7 : [255, 97, 107],
        8 : [10, 33, 227],
        9 : [255, 173, 243]
    }
    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        
        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255
        
        # add a scatter plot with the correponding color and label

        ax.scatter(current_tx, current_ty, c=color, label=label)

        # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    plt.show()
    plt.savefig('visualize_tsne_points.png')

def visualize_tsne(tsne, labels, plot_size=1000, max_image_size=100):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points(tx, ty, labels)

def load_pretrained_weights(model, args):
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        msg = model.load_state_dict(state_dict, strict=False)
        # print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("There is no reference weights available for this model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Visualizer TSNE')
    parser.add_argument('--pretrained_weights', default='./checkpoint/last.pth', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--lr", default=1e-1, type=float)
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for SGD")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum for SGD")
    parser.add_argument("--batch_size", default=128, type=int, help="On the contrastive step this will be multiplied by two.")
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers for Dataloader")
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    model = get_model()
    load_pretrained_weights(model, args)
    model.head = nn.Identity()
    model.eval()

    features, labels = get_features(model, args)
    tsne = TSNE(n_components=2).fit_transform(features)
    visualize_tsne(tsne, labels)

    