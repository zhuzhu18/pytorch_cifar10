import shutil
import torch
import os.path as osp

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.div_(batch_size))
    return res

def error(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.unsqueeze(1)).float()

    res = []
    for k in topk:
        correct_k = correct[:, :k].sum() / batch_size
        res.append(1 - correct_k)
    return res


def save_checkpoint(state, is_best, folder, filename='checkpoint.pth.tar'):
    fname = osp.join(folder, filename)
    torch.save(state, fname)
    if is_best:
        shutil.copyfile(fname, osp.join(folder, 'model_best.pth.tar'))


