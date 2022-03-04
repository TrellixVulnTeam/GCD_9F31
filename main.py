import argparse
import time
import os
import sys
from sklearn.metrics import log_loss
import torch

from torch.backends import cudnn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from utils import save_model, accuracy
from utils import AverageMeter
from loss.spc import ContrastiveLoss
from loss.NTXent import NTXent
from data.cifarloader import CIFAR100SampledSetLoaderMix, CIFAR10SampledSetLoaderMix
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
    parser.add_argument("--weight_coeff", default=0.35, type=float, help="Loss Weight Coeff")
    parser.add_argument("--checkpoint", default="./checkpoint/", type=str, help="Checkpoint folder")
    parser.add_argument("--boardlogs", default="logs", type=str, help="Tensorboard logs folder")

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

    return model

def train(model, train_loader, criterion, optimizer, writer, epoch, args):
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
        optimizer.zero_grad()

        features = model.head.forward(model(images))
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        output = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        u_loss = criterion(output)
        s_loss = criterion(output, labels)

        loss = ((1 - args.weight_coeff) * u_loss) + (args.weight_coeff * s_loss)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        end = time.time()
    
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

@torch.no_grad()
def validate_network(model, dataloader, criterion, args):
    acc_1_m = AverageMeter()
    acc_5_m = AverageMeter()
    losses = AverageMeter()
    model.eval()
    for batch_idx, (images, labels, idx) in enumerate(tqdm(dataloader)):
        images = torch.cat([images[0], images[1]], dim=0)
        images, labels = images.to(args.device), labels.to(args.device)
        bsz = labels.shape[0]
        features = model.head.forward(model(images))
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        output = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        u_loss = criterion(output)
        s_loss = criterion(output, labels)

        loss = ((1 - args.weight_coeff) * u_loss) + (args.weight_coeff * s_loss)

        # update metric
        losses.update(loss.item(), bsz)

        labels = torch.cat([labels, labels], dim=0)

        acc1, acc5 = accuracy(features, labels, topk=(1, 5))

        acc_1_m.update(acc1.item(), bsz)
        acc_5_m.update(acc5.item(), bsz)

    return acc_1_m.avg, acc_5_m.avg, losses.avg

def validate(model, valid_loader, args):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels, idx) in enumerate(tqdm(valid_loader)):
            x, labels = images[1].to(args.device), labels.to(args.device)
            output = model.head.forward(model(x))
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # Save checkpoint.
    acc = 100.0 * correct / total
    if acc > args.best_acc:
        args.best_acc = acc


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device


    # build data loader unsupervised
    if args.dataset == "cifar100":
        train_loader, val_loader = CIFAR100SampledSetLoaderMix(root=args.dataset_root, 
                                                                                        batch_size=args.batch_size, 
                                                                                        aug='twice', 
                                                                                        num_workers=args.num_workers,
                                                                                        shuffle=True)
        num_classes =  args.num_labeled_classes + args.num_unlabeled_classes
    else:
        train_loader, val_loader = CIFAR10SampledSetLoaderMix(root=args.dataset_root, 
                                                                                        batch_size=args.batch_size, 
                                                                                        aug='twice', 
                                                                                        num_workers=args.num_workers,
                                                                                        shuffle=True)
        num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    # load dino model
    model = get_model()
    
    cudnn.benchmark = True
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if name.startswith('blocks.11'):
            param.requires_grad = True
    model.head = DINOHead(in_dim=model.num_features, out_dim=num_classes)
    model = model.to(device)

    # for p in filter(lambda p: p.requires_grad, model.parameters()):
    #     print(p.requires_grad)

    # for name, child in model.named_children():
    #     print("--" + name + "--")
    #     for na, child in child.named_children():
    #         for param in child.parameters():
    #             print("{} {} ".format(na, param.requires_grad))


    if not os.path.isdir(args.boardlogs):
        os.makedirs(args.boardlogs)

    if not os.path.isdir(args.checkpoint):
        os.mkdir(args.checkpoint)

    writer = SummaryWriter(args.boardlogs)
    
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    
    args.best_acc = 0.0
    criterion = ContrastiveLoss(temperature=args.temperature)
    criterion.to(args.device)
  
    #decay schedule
    exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)

    #train
    for epoch in range(1, args.epochs + 1):
        time1 = time.time()
        loss = train(model, train_loader, criterion, optimizer, writer, epoch, args)
        acc1, acc5, ev_loss = validate_network(model, val_loader, criterion, args)
        # validate(model, val_loader, args)
        time2 = time.time()

        exp_lr_scheduler.step()

        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        writer.add_scalar("Model Loss | Epoch", loss, epoch)

        writer.add_scalar("Evaluation Loss | Epoch", ev_loss, epoch)

        # writer.add_scalar("Accuracy | Epoch", args.best_acc, epoch)

        writer.add_scalar("Accuracy Top 5 | Epoch", acc5, epoch)

        writer.add_scalar("Accuracy Top 1 | Epoch", acc1, epoch)

        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                args.checkpoint, 'ckpt_epoch_{epoch}_{dataset}.pth'.format(epoch=epoch, dataset=args.dataset))
            save_model(model, optimizer, args, epoch, save_file)

    writer.flush()
    writer.close()
    # save the last model
    save_file = os.path.join(
        args.checkpoint, '{dataset}_last.pth'.format(dataset=args.dataset))
    save_model(model, optimizer, args, args.epochs, save_file)
   
if __name__ == "__main__":
    main()