import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import fastervit
import warnings

torch.serialization.add_safe_globals([argparse.Namespace])
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
from peft import LoraConfig, get_peft_model


"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.block = block
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def resnet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


def LoraVIT(num_classes, r, head_freeze=True):
    model = fastervit.create_model(
        "faster_vit_0_224",
        pretrained=True,
        model_path="/tmp/faster_vit_0.pth.tar",
    )

    LoraConf = LoraConfig(
        r=r,
        lora_alpha=16,
        target_modules=["qkv"],
        lora_dropout=0.1,
        bias="none",
    )

    peft_model = get_peft_model(model, LoraConf)
    if head_freeze:
        peft_model.head = torch.nn.Linear(peft_model.head.weight.shape[-1], num_classes)
    else:
        peft_model.base_model.model.head = torch.nn.Linear(
            peft_model.base_model.model.head.in_features, num_classes
        )

    return peft_model


def net_weights_norm(net):
    net.eval()
    total_norm = 0.0
    for name_param, param in net.named_parameters():
        if param.requires_grad:  # Consider only trainable parameters
            total_norm += torch.norm(param, p=2).item() ** 2  # Compute squared L2 norm

    total_norm = total_norm**0.5  # Take the square root to get the L2 norm
    return total_norm


def net_dict_weights_norm(net_dict):
    # net_dict - OrderedDict() without buffers
    total_norm = 0.0

    for k, v in net_dict.items():
        if v.dtype.is_floating_point:
            norm = torch.norm(v, p=2).item() ** 2
            total_norm += norm

    total_norm = total_norm**0.5
    return total_norm


def summ_dicts(dict1, dict2, params):
    res_dict = copy.deepcopy(dict1)
    for param_name in params:
        if res_dict[param_name] is None:
            res_dict[param_name] = dict2[param_name].to("cpu")
        else:
            res_dict[param_name] = res_dict[param_name].to("cpu") + dict2[
                param_name
            ].to("cpu")
    return res_dict


def square_diff_dicts(dict1, dict2, params):
    res_dict = copy.deepcopy(dict1)
    for param_name in params:
        if res_dict[param_name] is None:
            res_dict[param_name] = dict2[param_name].to("cpu")
        else:
            res_dict[param_name] = (
                res_dict[param_name].to("cpu") - dict2[param_name].to("cpu")
            ) ** 2
    return res_dict


def diff_dicts(dict1, dict2, params):
    res_dict = copy.deepcopy(dict1)
    for param_name in params:
        if res_dict[param_name] is None:
            res_dict[param_name] = dict2[param_name].to("cpu")
        else:
            res_dict[param_name] = res_dict[param_name].to("cpu") - dict2[
                param_name
            ].to("cpu")
    return res_dict
