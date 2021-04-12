# coding: UTF-8


import torch
from torchsummary import summary
from .ISTA_Block import ISTA_Block
from .layers import ISTA_Loss


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ISTA(torch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, feature_num=64, layer_num=4, mode='normal', **kwargs):
        super(ISTA, self).__init__()

        self.input_conv = torch.nn.Conv2d(input_ch, output_ch, 3, 1, 1)
        self.layers = torch.nn.ModuleList([ISTA_Block(output_ch, output_ch, mode=mode) for _ in range(layer_num)])

    def forward(self, x):

        b, c, h, w = x.shape
        layer_sym = []
        x = self.input_conv(x)
        for i, layer in enumerate(self.layers):
            x, loss = layer(x)
            layer_sym.append(loss)
        layer_sym = torch.stack(layer_sym, dim=0).view(b, -1)

        return x, layer_sym


if __name__ == '__main__':

    model = ISTA(1, 31).to(device)
    summary(model, (1, 64, 64))
    x = torch.randn((3, 1, 64, 64)).to(device)
    xx = torch.randn(3, 31, 64, 64).to(device)
    y, sym_loss = model(x)
    loss_fn = ISTA_Loss().to(device)
    loss = loss_fn(xx, y, sym_loss)
