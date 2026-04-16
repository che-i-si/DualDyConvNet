import torch
from torch import nn


def conv_L(in_len, kernel, stride, padding=0):
    return int((in_len - kernel + 2 * padding) / stride + 1)


def cal_cnn_outlen(modules, in_len, pos):
    conv_l = in_len
    if isinstance(modules, nn.Sequential):
        for m in modules:
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                k = m.kernel_size[pos] if isinstance(m.kernel_size, tuple) else m.kernel_size
                s  = m.stride[pos] if isinstance(m.stride, tuple) else m.stride
                p = m.padding[pos] if isinstance(m.padding, tuple) else m.padding
                conv_l = conv_L(in_len, k, s, p)
                in_len = conv_l
            if isinstance(m, nn.AvgPool1d) or isinstance(m, nn.MaxPool1d):
                conv_l = conv_L(in_len, m.kernel_size, m.stride, m.padding)
                in_len = conv_l
            if isinstance(m, nn.AvgPool2d) or isinstance(m, nn.MaxPool2d):
                k = m.kernel_size[pos] if isinstance(m.kernel_size, tuple) else m.kernel_size
                s  = m.stride[pos] if isinstance(m.stride, tuple) else m.stride
                p = m.padding[pos] if isinstance(m.padding, tuple) else m.padding
                conv_l = conv_L(in_len, k, s, p)
                in_len = conv_l
    elif isinstance(modules, nn.ModuleList):
        for layer in modules:
            for m in layer:
                if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                    k = m.kernel_size[pos] if isinstance(m.kernel_size, tuple) else m.kernel_size
                    s  = m.stride[pos] if isinstance(m.stride, tuple) else m.stride
                    p = m.padding[pos] if isinstance(m.padding, tuple) else m.padding
                    conv_l = conv_L(in_len, k, s, p)
                    in_len = conv_l
                if isinstance(m, nn.AvgPool1d) or isinstance(m, nn.MaxPool1d):
                    conv_l = conv_L(in_len, m.kernel_size, m.stride, m.padding)
                    in_len = conv_l
                if isinstance(m, nn.AvgPool2d) or isinstance(m, nn.MaxPool2d):
                    k = m.kernel_size[pos] if isinstance(m.kernel_size, tuple) else m.kernel_size
                    s  = m.stride[pos] if isinstance(m.stride, tuple) else m.stride
                    p = m.padding[pos] if isinstance(m.padding, tuple) else m.padding
                    conv_l = conv_L(in_len, k, s, p)
                    in_len = conv_l

    else:
        if isinstance(modules, nn.Conv1d) or isinstance(modules, nn.Conv2d) or isinstance(modules, nn.Conv3d):
            k = modules.kernel_size[pos] if isinstance(modules.kernel_size, tuple) else modules.kernel_size
            s  = modules.stride[pos] if isinstance(modules.stride, tuple) else modules.stride
            p = modules.padding[pos] if isinstance(modules.padding, tuple) else modules.padding
            conv_l = conv_L(in_len, k, s, p)
        if isinstance(modules, nn.AvgPool1d) or isinstance(modules, nn.MaxPool1d):
            conv_l = conv_L(in_len, modules.kernel_size, modules.stride, modules.padding)
        if isinstance(modules, nn.AvgPool2d) or isinstance(modules, nn.MaxPool2d):
            k = modules.kernel_size[pos] if isinstance(modules.kernel_size, tuple) else modules.kernel_size
            s  = modules.stride[pos] if isinstance(modules.stride, tuple) else modules.stride
            p = modules.padding[pos] if isinstance(modules.padding, tuple) else modules.padding
            conv_l = conv_L(in_len, k, s, p)
    return conv_l


def get_bias(block):
    params = []
    for modules in block.modules():
        if isinstance(modules, nn.Conv2d) or isinstance(modules, nn.Linear):
            params.append(modules.bias)
    return params

