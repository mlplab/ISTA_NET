# coding: utf-8


import torch
from torchsummary import summary
from .layers import My_HSI_network, swish, mish, RAM, HSI_prior_block


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DW_SP_Model(torch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, feature=64, block_num=9, **kwargs):
        super(DW_SP_Model, self).__init__()

        # keys = kwargs.keys()
        self.activation = None
        self.output_norm = None
        if 'activation' in kwargs:
            self.activation = kwargs['activation']
        if 'output_norm' in kwargs:
            self.output_norm = kwargs['output_norm']
        self.start_conv = torch.nn.Conv2d(input_ch, output_ch, 3, 1, 1)
        self.start_shortcut = torch.nn.Identity()
        hsi_block = []
        residual_block = []
        shortcut_block = []
        for _ in range(block_num):
            hsi_block.append(My_HSI_network(output_ch, output_ch))
            residual_block.append(torch.nn.Conv2d(output_ch, output_ch, 3, 1, 1))
            shortcut_block.append(torch.nn.Identity())
        self.hsi_block = torch.nn.Sequential(*hsi_block)
        self.residual_block = torch.nn.Sequential(*residual_block)
        self.shortcut_block = torch.nn.Sequential(*shortcut_block)
        self.output_conv = torch.nn.Conv2d(output_ch, output_ch, 3, 1, 1)

    def forward(self, x):

        x = self.start_conv(x)
        h = self.start_shortcut(x)
        for hsi_block, residual_block, shortcut_block in zip(self.hsi_block, self.residual_block, self.shortcut_block):
            x_hsi = hsi_block(x)
            x_res = residual_block(x)
            x_start = shortcut_block(h)
            x = x_res + x_start + x_hsi
        return self._output_norm_fn(self.output_conv(x))

    def _activation_fn(self, x):
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)
        else:
            return x

    def _output_norm_fn(self, x):
        if self.output_norm == 'sigmoid':
            return torch.sigmoid(x)
        elif self.output_norm == 'tanh':
            return torch.tanh(x)
        else:
            return x


class Attention_HSI_Model(torch.nn.Module):

    def __init__(self, input_ch, output_ch, feature=64, block_num=9, activation='relu', output_norm=None):
        super(Attention_HSI_Model, self).__init__()
        self.activation = activation
        self.output_norm = output_norm
        self.start_conv = torch.nn.Conv2d(input_ch, output_ch, 1, 1, 0)
        self.start_shortcut = torch.nn.Identity()
        hsi_block = []
        residual_block = []
        shortcut = []
        for _ in range(block_num):
            hsi_block.append(HSI_prior_block(output_ch, output_ch, feature=feature))
            # residual_block.append(torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0))
            residual_block.append(RAM(output_ch, output_ch, ratio=4))
            shortcut.append(torch.nn.Identity())
        self.hsi_block = torch.nn.Sequential(*hsi_block)
        self.residual_block = torch.nn.Sequential(*residual_block)
        self.shortcut = torch.nn.Sequential(*shortcut)
        self.output_conv = torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)

    def forward(self, x):
        x = self.start_conv(x)
        h = self.start_shortcut(x)
        for hsi_block, residual_block, shortcut in zip(self.hsi_block, self.residual_block, self.shortcut):
            x_hsi = hsi_block(x)
            x_res = residual_block(x)
            x_shortcut = shortcut(h)
            x = x_res + x_shortcut + x_hsi
        return self._output_norm_fn(self.output_conv(x))

    def _activation_fn(self, x):
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)

    def _output_norm_fn(self, x):
        if self.output_norm == 'sigmoid':
            return torch.sigmoid(x)
        elif self.output_norm == 'tanh':
            return torch.tanh(x)
        else:
            return x


if __name__ == '__main__':

    model = DW_SP_Model(1, 31, output_norm=None).to(device)
    summary(model, (1, 64, 64))
    model = Attention_HSI_Model(1, 31, output_norm=None).to(device)
    summary(model, (1, 64, 64))
