import argparse
import os
import random
import shutil
import time
import warnings
import pandas as pd
import numpy
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter

from model import levi, levi_bn, levi_2layer, mm_cls

CustomModel = {'levi':levi,'levi_bn':levi_bn,'levi_2layer':levi_2layer}

class CustomImageDataset(Dataset):
    def __init__(self, img_name_file, label_file, img_dir, transform=None, target_transform=None):
        self.img_names = pd.read_csv(img_name_file)
        self.img_labels = pd.read_csv(label_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names.iloc[idx]['img_path'])
        image = Image.open(img_path)
        age_embed = torch.tensor([self.img_names.iloc[idx]['age_embed'],self.img_names.iloc[idx]['is_baby']])
        label = self.img_labels.iloc[idx]['gender']
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, age_embed, label



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])) + [name for name in CustomModel]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='levi',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: levi)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=50, type=int,
                    metavar='N',
                    help='mini-batch size (default: 50), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--age', default=False, type=bool, 
                    help='enable age embedding (default: False)')
parser.add_argument('--crop', default='center', type=str, 
                    help='image crop center|random|five_f1|five_f2')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--optim', default='sgd', type=str, 
                    help='optimizer sgd|adam|adamw')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                    metavar='W', help='weight decay (default: 1e-2)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=40, type=int,
                    metavar='N', help='print frequency (default: 40)')
parser.add_argument('--log', default=True, type=bool,
                    help='tensorboard log (default: True)')
parser.add_argument('--logsuf', default='', type=str,
                    help='tensorboard logdir name suffix (default: "")')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    inplanes = 15 if args.crop == 'five_f2' else 3

    if args.arch.startswith('levi'):
        out_channels, size, shape = 512, 256, 227
        backbone = CustomModel[args.arch]

    elif args.arch.startswith('resnet'):
        out_channels, size, shape = 1000, 256, 224
        backbone = models.__dict__[args.arch]

    elif args.arch.startswith('inception'):
        out_channels, size, shape = 1000, 342, 299
        backbone = models.__dict__[args.arch]

    else:
        print("not supported backbone arch!")
        return

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        backbone = backbone(pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch not in models.__dict__:
            backbone = backbone(inplanes=inplanes)
        else:
            backbone = backbone()

    if args.age is True:
        print("=> enabled age embedding")
        model = mm_cls(backbone, out_channels=out_channels, embed_dim=2)
    else:
        model = mm_cls(backbone, out_channels=out_channels, embed_dim=0)

    print(model)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code


    if args.crop == 'center':
        Crop = transforms.CenterCrop
    elif args.crop == 'random':
        Crop = transforms.RandomCrop

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.crop.startswith('five'):
        t = transforms.Compose([
                    transforms.Resize(size),
                    transforms.FiveCrop(shape),
                    transforms.Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
                    normalize,
             ])
    else:
        t = transforms.Compose([
                    transforms.Resize(size),
                    Crop(shape),
                    transforms.ToTensor(),
                    normalize,
             ])
    train_dataset = CustomImageDataset('X_train_age.csv','y_train.csv',args.data,t)
    val_dataset = CustomImageDataset('X_test_age.csv','y_test.csv',args.data,t)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.log:
        import socket
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(
            'runs', args.arch + args.logsuf +' '+current_time + '_' + socket.gethostname())
        writer = SummaryWriter(log_dir=log_dir)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    best_epoch = 1

    for epoch in range(args.start_epoch+1, args.epochs+1):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        loss_t, acc_t = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        loss_v, acc_v = validate(val_loader, model, criterion, args)

        if args.log:
            # update log
            writer.add_scalar('Loss/train', loss_t, epoch)
            writer.add_scalar('Loss/test', loss_v, epoch)
            writer.add_scalar('Accuracy/train', acc_t, epoch)
            writer.add_scalar('Accuracy/test', acc_v, epoch)

        # remember best acc@1 and save checkpoint
        is_best = acc_v > best_acc1
        if is_best:
            best_acc1 = acc_v
            best_epoch = epoch
        print(' * Acc@1 {:.3f} (Best {:.3f} @epoch{})'.format(acc_v,best_acc1,best_epoch))
        print('...')

        save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.3e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    #top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, age_embed, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            age_embed = age_embed.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        #five crop input processing
        if args.crop == 'five_f1':
            bs, ncrops, c, h, w = images.size()
            images = images.view(-1,c,h,w)
        if args.crop == 'five_f2':
            bs, ncrops, c, h, w = images.size()
            images = images.view(bs,-1,h,w)

        # compute output
        output = model([images,age_embed])

        if args.arch.startswith('inception'):
            output = output[0]
        if args.crop == 'five_f1':
            output = output.view(bs,ncrops,-1).mean(1)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        #top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.3e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    #top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, age_embed, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                age_embed = age_embed.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            #five crop input processing
            if args.crop == 'five_f1':
                bs, ncrops, c, h, w = images.size()
                images = images.view(-1,c,h,w)
            if args.crop == 'five_f2':
                bs, ncrops, c, h, w = images.size()
                images = images.view(bs,-1,h,w)

            # compute output
            output = model([images,age_embed])
            if args.crop == 'five_f1':
                output = output.view(bs,ncrops,-1).mean(1)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            #top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    return losses.avg, top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    lr = args.lr * (0.1 ** (epoch // 40))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size)[0])
        return res

if __name__ == '__main__':
    main()

