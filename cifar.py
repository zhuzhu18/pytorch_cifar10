from options.opt_cifar import get_configs
from datasets import CIFAR10, CIFAR100
from trainers.train import ClassificationTrainer
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
import os
import os.path as osp
import setproctitle

import models.preresnet as models

def main():
    global best_prec1, train_rec, test_rec
    conf = get_configs()

    conf.root = "work"
    conf.folder = osp.join(conf.root, conf.arch)
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    setproctitle.setproctitle(conf.arch)

    os.makedirs(conf.folder, exist_ok=True)

    if conf.dataset == "cifar10":
        CIFAR = CIFAR10(conf.data)    # Datasets object
    else:
        CIFAR = CIFAR100(conf.data)

    # create model
    if conf.pretrained:
        print("=> using pre-trained model '{}'".format(conf.arch))
        raise NotImplementedError("pre-trained is not supported on CIFAR")
        # model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(conf.arch))
        model = models.__dict__[conf.arch](CIFAR.num_classes)
    # print(model.features)
    conf.distributed = conf.distributed_processes > 1
    if not conf.distributed:
        if conf.gpus > 0:
            model = nn.DataParallel(model)
        model.to(conf.device)
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), conf.lr, momentum=conf.momentum, weight_decay=conf.weight_decay)

    # optionally resume from a checkpoint
    if conf.resume:
        if os.path.isfile(conf.resume):
            print("=> loading checkpoint from '{}'".format(conf.resume))
            checkpoint = torch.load(conf.resume)
            conf.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint done. (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(conf.resume))

    cudnn.benchmark = True  # improve the efficiency of the program

    train_loader, valid_loader = CIFAR.get_loader(conf)
    trainer = ClassificationTrainer(model, criterion, conf, optimizer)

    if conf.evaluate:
        trainer.evaluate(valid_loader, model, criterion)
        return

    step1 = int(conf.epochs * 0.5)
    step2 = int(conf.epochs * 0.75)
    lr_scheduler = MultiStepLR(optimizer, milestones=[step1, step2], gamma=0.1)

    trainer.fit(train_loader, valid_loader, start_epoch=0, max_epochs=200,
                lr_scheduler=lr_scheduler)

if __name__ == '__main__':
    main()
