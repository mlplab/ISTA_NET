# coding: utf-8


import torch
from torchsummary import summary
from .layers import Ghost_layer, Ghost_Bottleneck, ReLU, Leaky, Swish, Mish, FReLU


class Ghost_HSCNN(torch.nn.Module):

    def __init__(self, input_ch, output_ch, feature_num=64, layer_num=9, ratio=2, **kwargs):
        super(Ghost_HSCNN, self).__init__()
        activation = kwargs.get('activation', 'relu').lower()
        mode = kwargs.get('mode', 'None').lower()
        activations = {'relu': ReLU, 'leaky': Leaky, 'swish': Swish, 'mish': Mish, 'frelu': FReLU}
        activation_kernel =  kwargs.get('activation_kernel', 3)
        activation_stride =  kwargs.get('activation_stride', 1)
        self.start_conv = torch.nn.Conv2d(input_ch, feature_num, 3, 1, 1)
        self.ghost_layers = torch.nn.ModuleList([Ghost_layer(feature_num, feature_num, ratio=ratio, mode=mode) for _ in range(layer_num)])
        self.activations = torch.nn.ModuleList([activations[activation](feature_num, feature_num, activation_kernel, activation_stride) for _ in range(layer_num)])
        self.output_conv = torch.nn.Conv2d(feature_num, output_ch, 3, 1, 1)

    def forward(self, x):

        x = self.start_conv(x)
        x_in = x
        for ghost_layer, activation in zip(self.ghost_layers, self.activations):
            x = activation(ghost_layer(x))
        output = self.output_conv(x + x_in)
        return output

    def show_features(self, x, layer_num=0, output_layer=True, activation=True):

        # initialize result
        result = []
        if isinstance(layer_num, int):
            layer_num = [layer_num]
        layer_num = set(layer_num)
        layer_nums = []
        layer_nums = [True if i in layer_num else False for i in range(len(self.ghost_layers))]

        # add start_conv
        x = self.start_conv(x)
        if layer_nums[0]:
            result.append(x)
        x_in = x

        # add feature map
        for i, feature_map in enumerate(self.ghost_layers):
            x = feature_map(x)
            # if activation is True:
            x = self._activation_fn(x)
            if layer_nums[i]:
                result.append(x)

        output = self.output_conv(x + x_in)
        if output_layer:
            result.append(output)
        return result


class Ghost_HSCNN_Bneck(torch.nn.Module):

    def __init__(self, input_ch, output_ch, feature_num=64, layer_num=9, **kwargs):
        super(Ghost_HSCNN_Bneck, self).__init__()
        self.activation = kwargs.get('activation', 'relu')
        ratio = kwargs.get('ratio', 2)
        se_flag = kwargs.get('se_flag', False)
        self.start_conv = torch.nn.Conv2d(input_ch, feature_num, 3, 1, 1)
        self.ghost_layers = torch.nn.ModuleList([Ghost_Bottleneck(feature_num, feature_num // ratio, feature_num, se_flag=se_flag) for _ in range(layer_num)])
        self.output_conv = torch.nn.Conv2d(feature_num, output_ch, 3, 1, 1)

    def _activation_fn(self, x):
        if self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)
        elif self.activation == 'leaky' or self.activation == 'leaky_relu':
            return torch.nn.functional.leaky_relu(x)
        elif self.activation == 'relu':
            return torch.relu(x)
        else:
            return x

    def forward(self, x):

        x = self.start_conv(x)
        x_in = x
        for ghost_layer in self.ghost_layers:
            x = self._activation_fn(ghost_layer(x))
        output = self.output_conv(x)
        return output



if __name__ == '__main__':

    x = torch.rand((1, 1, 64, 64))
    # model = Ghost_HSCNN_Bneck(1, 31, ratio=2)
    summary(model, (1, 64, 64))
    model = Ghost_HSCNN(1, 31)
    summary(model, (1, 64, 64))
    result = model.show_features(x, layer_num=list(range(1, 9 + 1)), output_layer=True)
    print(result[0].shape)
