import argparse
import time
import os
import sys
import torch

from torch.backends import cudnn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from utils import save_model
from utils import AverageMeter
from loss.spc import SupervisedContrastiveLoss
from loss.sspc import SelfSupervisedContrastiveLoss
from data.cifarloader import CIFAR100SampledSetLoader, CIFAR10SampledSetLoader, CIFAR100Loader, CIFAR10Loader
import vision_transformer as vits
from vision_transformer import DINOHead
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=["cifar10", "cifar100"],
        help="dataset name",
    )
    parser.add_argument("--batch_size", default=128, type=int, help="On the contrastive step this will be multiplied by two.")
    parser.add_argument("--temperature", default=0.07, type=float, help="Constant for loss no thorough")
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr", default=1e-1, type=float)
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for SGD")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum for SGD")
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers for Dataloader")
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)
    parser.add_argument('--num_labeled_classes', default=5, type=int)
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--print_freq', type=int, default=5, help='print frequency')
    parser.add_argument("--loss_reg", default=0.35, type=float, help="Loss regularization")

    args = parser.parse_args()

    return args

def get_model(pretrained=True, **kwargs):
    model = vits.__dict__["vit_base"](patch_size=16, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)

    for param in model.parameters():
        param.requires_grad = False
    model.norm.bias.requires_grad = True
    model.norm.weight.requires_grad = True
    for name, param in model.named_parameters():
        if name.startswith('blocks.11'):
            param.requires_grad = True
    
    return model

def train_supervised(model, train_loader, eval_loader, criterion, optimizer, writer, epoch, args):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    
    for batch_idx, (images, labels, idx) in enumerate(tqdm(train_loader)):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        images, labels = images.to(args.device), labels.to(args.device)
        bsz = labels.shape[0]
        # compute loss
        features = model.head.forward(model(images))
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, labels)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        writer.add_scalar(
            "Loss train | Supervised Contrastive",
            loss.item(),
            epoch * len(train_loader) + batch_idx,
        )
    
        # print info
        if (batch_idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, batch_idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    
    return losses.avg

def train_unsupervised(model, train_loader, criterion, optimizer, writer, epoch, args):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    
    for batch_idx, (images, labels, idx) in enumerate(tqdm(train_loader)):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        images, labels = images.to(args.device), labels.to(args.device)
        bsz = labels.shape[0]
        # compute loss
        features = model.head.forward(model(images))
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        writer.add_scalar(
            "Loss train | Unsupervised Contrastive",
            loss.item(),
            epoch * len(train_loader) + batch_idx,
        )
    
        # print info
        if (batch_idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, batch_idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    
    return losses.avg
    

def test(epoch, model, test_loader, criterion, writer, args):
    """
    :param epoch: int
    :param model: torch.nn.Module, Model
    :param test_loader: torch.utils.data.DataLoader
    :param criterion: torch.nn.Module, Loss
    :param writer: torch.utils.tensorboard.SummaryWriter
    :param args: argparse.Namespace
    :return:
    """

    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (x, label, idx) in enumerate(tqdm(test_loader)):
            x, label = x.to(args.device), label.to(args.device)
            output = model(x)
            loss = criterion(output, label)

            test_loss += loss.item()
            _, predicted = output.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            progress_bar(
                batch_idx,
                len(test_loader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    # Save checkpoint.
    acc = 100.0 * correct / total
    writer.add_scalar("Accuracy validation | Cross Entropy", acc, epoch)

    if acc > args.best_acc:
        # print("Saving..")
        # state = {
        #     "net": model.state_dict(),
        #     "acc": acc,
        #     "epoch": epoch,
        # }
        # if not os.path.isdir("checkpoint"):
        #     os.mkdir("checkpoint")
        # torch.save(state, "./checkpoint/ckpt_cross_entropy.pth")
        args.best_acc = acc


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    # build data loader supervised
    if args.dataset == "cifar100":
        s_train_loader, s_valid_loader, s_test_loader = CIFAR100SampledSetLoader(root=args.dataset_root, 
                                                                                batch_size=args.batch_size, 
                                                                                aug='twice', 
                                                                                num_workers=args.num_workers,
                                                                                shuffle=True)
        num_classes =  args.num_labeled_classes + args.num_unlabeled_classes
    else:
        s_train_loader, s_valid_loader, s_test_loader = CIFAR10SampledSetLoader(root=args.dataset_root, 
                                                                                batch_size=args.batch_size, 
                                                                                aug='twice', 
                                                                                num_workers=args.num_workers, 
                                                                                shuffle=True)
        num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    # build data loader unsupervised
    if args.dataset == "cifar100":
        u_train_loader = CIFAR100Loader(root=args.dataset_root, 
                                        batch_size=args.batch_size, 
                                        aug='twice', 
                                        num_workers=args.num_workers,
                                        shuffle=True)
        num_classes =  args.num_labeled_classes + args.num_unlabeled_classes
    else:
        u_train_loader = CIFAR10Loader(root=args.dataset_root, 
                                        batch_size=args.batch_size, 
                                        aug='twice', 
                                        num_workers=args.num_workers,
                                        shuffle=True)
        num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    # load dino model
    model = get_model()
    model.head = DINOHead(in_dim=model.num_features, out_dim=num_classes)
    model = model.to(device)
    cudnn.benchmark = True

    if not os.path.isdir("logs"):
        os.makedirs("logs")

    if not os.path.isdir("checkpoint"):
        os.mkdir("checkpoint")

    s_writer = SummaryWriter("logs")
    u_writer = SummaryWriter("logs")
    g_writer = SummaryWriter("logs")
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    
    args.best_acc = 0.0
    s_criterion = SupervisedContrastiveLoss(temperature=args.temperature)
    u_criterion = SelfSupervisedContrastiveLoss(temperature=args.temperature)
    s_criterion.to(args.device)
    u_criterion.to(args.device)

    print(len(u_train_loader))

    #train
    for epoch in range(1, args.epochs + 1):
        #decay schedule
        exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)

        time1 = time.time()
        # s_loss = train_supervised(model, s_train_loader, s_valid_loader, s_criterion, optimizer, s_writer, epoch, args)
        u_loss = train_unsupervised(model, u_train_loader, u_criterion, optimizer, u_writer, epoch, args)
        time2 = time.time()

        # total_loss = ((1 - args.loss_reg) * u_loss) + (args.loss_reg * s_loss)

        if epoch >= 10:
            exp_lr_scheduler.step()

        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # g_writer.add_scalar(
        #     "Loss train | Overall training",
        #     total_loss,
        #     epoch * (len(s_train_loader) + len(u_train_loader)) ,
        # )

        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                './checkpoint/', 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, args, epoch, save_file)


    # save the last model
    save_file = os.path.join(
        './checkpoint/', 'last.pth')
    save_model(model, optimizer, args, args.epochs, save_file)

if __name__ == "__main__":
    main()