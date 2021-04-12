# coding: UTF-8


import torch
from torchsummary import summary
from .layers import Ghost_Bottleneck


class Ghost_Reconst_Net(torch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, feature_num=64, layer_num=9, **kwargs):
        super(Ghost_Reconst_Net, self).__init__()


        se_flag = kwargs.get('se_flag', False)
        activation = kwargs.get('activation')
        if output_ch % 2 != 0:
            ghost_output = output_ch + 1
        else:
            ghost_output = output_ch

        self.start_conv = torch.nn.Conv2d(input_ch, output_ch, 3, 1, 1)
        self.ghost_conv = torch.nn.Conv2d(output_ch, ghost_output, 3, 1, 1)
        self.ghost_layers = torch.nn.ModuleList([Ghost_Bottleneck(ghost_output,
                                                                  feature_num,
                                                                  ghost_output,
                                                                  stride=1,
                                                                  activation=activation,
                                                                  se_flag=se_flag) for _ in range(layer_num)])
        self.spectral_layers = torch.nn.ModuleList([torch.nn.Conv2d(ghost_output, ghost_output, 1, 1, 0) for _ in range(layer_num)])
        # self.share_conv = torch.nn.Conv2d(ghost_output, ghost_output, 3, 1, 1)
        self.output_conv = torch.nn.Conv2d(ghost_output, output_ch, 1, 1, 0)

    def forward(self, x):
        h = self.start_conv(x)
        x = self.ghost_conv(h)
        for ghost_layer, spectral_layer in zip(self.ghost_layers, self.spectral_layers):
            x_shortcut = x
            x = ghost_layer(x)
            x = spectral_layer(x)
            # x_res = self.share_conv(x)
            x = x + x_shortcut
        x = self.output_conv(x) + h
        return x


if __name__ == '__main__':

    model = Ghost_Reconst_Net(1, 31)
    summary(model, input_size=(1, 48, 48))
