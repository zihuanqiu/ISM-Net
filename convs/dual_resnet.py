"""Taken & slightly modified from:
* https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import torch
import torch.nn as nn
from convs.resnet import resnet18
from convs.cifar_resnet import resnet20, resnet32


class DualExpert(nn.Module):
    def __init__(self, conv_type='resnet32'):
        super(DualExpert, self).__init__()
        assert conv_type in ['resnet18', 'resnet32', 'resnet20']
        if conv_type == 'resnet18':
            self.LearningExpert = resnet18()
            self.MemoryExpert = resnet18()
        elif conv_type == 'resnet32':
            self.LearningExpert = resnet32()
            self.MemoryExpert = resnet32()
        elif conv_type == 'resnet20':
            self.LearningExpert = resnet20()
            self.MemoryExpert = resnet20()

        self.feat_dim = 50
        self.memory_dim = 50
        self.change_dim(self.feat_dim, self.memory_dim)

    def forward(self, x):
        if self.feat_dim > 0:
            x_f = self.LearningExpert(x)['fmaps'][-1]
            x_f = self.finalBlock_f(x_f)
            x_f = x_f.mean(dim=[-1, -2])
        else:
            x_f = None

        if self.memory_dim > 0:
            x_s = self.MemoryExpert(x)['fmaps'][-1]
            x_s = self.finalBlock_s(x_s)
            x_s = x_s.mean(dim=[-1, -2])
        else:
            x_s = None

        return x_f, x_s

    def change_dim(self, feat_dim, memory_dim, device='cpu'):
        if self.feat_dim != feat_dim:
            self.feat_dim = feat_dim
        if self.memory_dim != memory_dim:
            self.memory_dim = memory_dim

        if memory_dim != 0:
            self.finalBlock_s = nn.Sequential(
                nn.Conv2d(self.MemoryExpert.inplanes, self.memory_dim, 1),
                nn.BatchNorm2d(self.memory_dim),
                nn.ReLU(inplace=True)).to(device)

        if feat_dim != 0:
            self.finalBlock_f = nn.Sequential(
                nn.Conv2d(self.LearningExpert.inplanes, self.feat_dim, 1),
                nn.BatchNorm2d(self.feat_dim),
                nn.ReLU(inplace=True)).to(device)


def dual_expert(conv_type='resnet32'):
    model = DualExpert(conv_type)
    return model


if __name__ == '__main__':
    input = torch.randn(3, 3, 32, 32)
    model = dual_expert()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    out = model(input)
    # print(out)