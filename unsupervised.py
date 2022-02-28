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

    for param in model.parameters():
        param.requires_grad = False
    model.norm.bias.requires_grad = True
    model.norm.weight.requires_grad = True
    for name, param in model.named_parameters():
        if name.startswith('blocks.11'):
            param.requires_grad = True
    
    return model

def train(model, train_loader, u_criterion, s_criterion, optimizer, writer, epoch, args):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    accuracy_list = list()

    end = time.time()
    
    for batch_idx, (images, labels, idx, labeled) in enumerate(tqdm(train_loader)):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        images, labels = images.to(args.device), labels.to(args.device)
        bsz = labels.shape[0]
        # compute loss
        features = model.head.forward(model(images))
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        output = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        u_loss = u_criterion(output)
        if labeled.numpy()[0]:
            s_loss = s_criterion(output, labels)
        else:
            s_loss = s_criterion(output, None)

        loss = ((1 - args.weight_coeff) * u_loss) +  (args.weight_coeff * s_loss)

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

# def test(model, dataloader, writer, epoch, args):
#     acc_record = AverageMeter()
#     model.eval()
#     for batch_idx, (data, label, idx) in enumerate(tqdm(dataloader)):
#         data, label = data.to(args.device), label.to(args.device)
#         output = model.head.forward(model(data))
#         # measure accuracy and record loss
#         acc = accuracy(output, label)
#         acc_record.update(acc[0].item(), data.size(0))
        

#     print('Test Acc: {:.4f}'.format(acc_record.avg))
#     return acc_record 

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device


    # build data loader unsupervised
    if args.dataset == "cifar100":
        s_labeled_loader, s_valid_loader, s_unlabeled_loader, mixed_loader = CIFAR100SampledSetLoader(root=args.dataset_root, 
                                                                                        batch_size=args.batch_size, 
                                                                                        aug='twice', 
                                                                                        num_workers=args.num_workers,
                                                                                        shuffle=True)
        num_classes =  args.num_labeled_classes + args.num_unlabeled_classes
    else:
        s_labeled_loader, s_valid_loader, s_unlabeled_loader, mixed_loader = CIFAR10SampledSetLoader(root=args.dataset_root, 
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

    if not os.path.isdir(args.boardlogs):
        os.makedirs(args.boardlogs)

    if not os.path.isdir(args.checkpoint):
        os.mkdir(args.checkpoint)

    writer = SummaryWriter(args.boardlogs)
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    
    args.best_acc = 0.0
    u_criterion = SelfSupervisedContrastiveLoss(temperature=args.temperature)
    u_criterion.to(args.device)
    s_criterion = SupervisedContrastiveLoss(temperature=args.temperature)
    s_criterion.to(args.device)

    #train
    for epoch in range(1, args.epochs + 1):
        #decay schedule
        exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)

        time1 = time.time()
        loss = train(model, mixed_loader, u_criterion, s_criterion, optimizer, writer, epoch, args)
        # acc = test(model, valid_loader, writer, epoch, args)
        time2 = time.time()

        exp_lr_scheduler.step()

        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        writer.add_scalar("Model Loss | Overall training", loss, epoch)

        # writer.add_scalar("Accuracy validation", acc.avg, epoch)

        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                args.checkpoint, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, args, epoch, save_file)

    writer.flush()
    writer.close()
    # save the last model
    save_file = os.path.join(
        args.checkpoint, 'last.pth')
    save_model(model, optimizer, args, args.epochs, save_file)
   
if __name__ == "__main__":
    main()