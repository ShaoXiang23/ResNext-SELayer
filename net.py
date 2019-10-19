import torch
from torchvision.models.resnet import _resnet
from se_block import SE_Bottleneck

def resnext50_32x4d(pretrained=False, progress=True, num_classes = 1000, **kwargs):
    """Constructs a ResNeXt-50 32x4d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    kwargs['num_classes'] = num_classes
    return _resnet('resnext50_32x4d', SE_Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, num_classes = 1000, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    kwargs['num_classes'] = num_classes
    return _resnet('resnext101_32x8d', SE_Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
