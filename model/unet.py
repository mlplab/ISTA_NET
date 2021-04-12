# coding: utf-8


import torch
import torchvision
from torchsummary import summary
from base_model import Conv_Block_UNet, SA_Block


class UNet(torch.nn.Module):

    def __init__(self, input_ch, output_ch, feature_list=None):
        super(UNet, self).__init__()

        if feature_list is None:
            feature_list = [32, 64, 128, 256, 512, 1024]
        layers_num = [3, 3, 3, 3, 2, 2]
        decode_layers_num = layers_num[::-1]
        self.pool_flag = [True for _ in range(len(feature_list) - 1)]
        self.pool_flag[0] = False
        self.start_conv = Conv_Block_UNet(input_ch, feature_list[0], 3, 1, 1)
        encoder = []
        for i in range(len(feature_list) - 1):
            conv_layer = []
            conv_layer.append(Conv_Block_UNet(feature_list[i], feature_list[i + 1], 3, 1, 1))
            for _ in range(layers_num[i + 1]):
                conv_layer.append(Conv_Block_UNet(feature_list[i + 1], feature_list[i + 1], 3, 1, 1, norm=True))
            encoder.append(torch.nn.Sequential(*conv_layer))
        self.encoder = torch.nn.Sequential(*encoder)

        decode_feature_list = feature_list[:0:-1]
        decode = []
        up_block = []
        for i in range(len(decode_feature_list) - 1):
            up_block.append(torch.nn.ConvTranspose2d(feature_list[-(i + 1)], feature_list[-(i + 1)] // 2, kernel_size=2, stride=2)
                            )
            conv_layer = [Conv_Block_UNet(feature_list[-(i + 1)], decode_feature_list[i + 1], 3, 1, 1)]
            for _ in range(decode_layers_num[i + 1] - 1):
                conv_layer.append(Conv_Block_UNet(decode_feature_list[i + 1], decode_feature_list[i + 1], 3, 1, 1, norm=False))
            decode.append(torch.nn.Sequential(*conv_layer))
        self.up_block = torch.nn.Sequential(*up_block)
        self.decoder = torch.nn.Sequential(*decode)
        self.last_conv = Conv_Block_UNet(
            decode_feature_list[-1], feature_list[0], 3, 1, 1)
        self.output = torch.nn.Conv2d(feature_list[0], output_ch, 1, 1, 0)

    def forward(self, x):

        x = self.start_conv(x)
        encode = []
        for i in range(len(self.encoder) - 1):
            if self.pool_flag[i] is True:
                x = torch.nn.functional.max_pool2d(x, 2)
            x = self.encoder[i](x)
            encode.append(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.encoder[-1](x)

        encode = encode[::-1]
        for i, decoder in enumerate(self.decoder):
            x = self.up_block[i](x)
            x = torch.cat([x, encode[i]], dim=1)
            x = decoder(x)
        x = self.last_conv(x)
        x = torch.tanh(self.output(x))
        return x


class UNet_none(torch.nn.Module):

    def __init__(self, input_ch, output_ch, feature_list=None):
        super(UNet_none, self).__init__()

        if feature_list is None:
            feature_list = [32, 64, 128, 256, 512, 1024]
        self.pool_flag = [True for _ in range(len(feature_list) - 1)]
        self.pool_flag[0] = False
        self.start_conv = Conv_Block_UNet(
            input_ch, feature_list[0], 3, 1, 1, norm=False)
        encoder = []
        for i in range(len(feature_list) - 1):
            conv_layer = []
            conv_layer.append(Conv_Block_UNet(feature_list[i], feature_list[i + 1], 3, 1, 1, norm=False))
            conv_layer.append(Conv_Block_UNet(feature_list[i + 1], feature_list[i + 1], 3, 1, 1, norm=False))
            encoder.append(torch.nn.Sequential(*conv_layer))
        self.encoder = torch.nn.Sequential(*encoder)

        decode_feature_list = feature_list[:0:-1]
        decode = []
        up_block = []
        for i in range(len(decode_feature_list) - 1):
            up_block.append(torch.nn.ConvTranspose2d(feature_list[-(i + 1)], feature_list[-(i + 1)] // 2, kernel_size=2, stride=2))
            conv_layer = [Conv_Block_UNet(feature_list[-(i + 1)], decode_feature_list[i + 1], 3, 1, 1, norm=False),
                          Conv_Block_UNet(decode_feature_list[i + 1], decode_feature_list[i + 1], 3, 1, 1, norm=False)]
            decode.append(torch.nn.Sequential(*conv_layer))
        self.up_block = torch.nn.Sequential(*up_block)
        self.decoder = torch.nn.Sequential(*decode)
        self.last_conv = Conv_Block_UNet(
            decode_feature_list[-1], feature_list[0], 3, 1, 1, norm=False)
        self.output = torch.nn.Conv2d(feature_list[0], output_ch, 1, 1, 0)

    def forward(self, x):

        x = self.start_conv(x)
        encode = []
        for i in range(len(self.encoder) - 1):
            if self.pool_flag[i] is True:
                x = torch.nn.functional.max_pool2d(x, 2)
            x = self.encoder[i](x)
            encode.append(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.encoder[-1](x)

        encode = encode[::-1]
        for i, decoder in enumerate(self.decoder):
            x = self.up_block[i](x)
            x = torch.cat([x, encode[i]], dim=1)
            x = decoder(x)
        x = self.last_conv(x)
        x = torch.tanh(self.output(x))
        return x


class UNet_Res(UNet):

    def forward(self, x):

        x_in = x
        x = self.start_conv(x)
        # x_in = self.start_shortcut(x)
        encode = []
        for i in range(len(self.encoder) - 1):
            if self.pool_flag[i] is True:
                x = torch.nn.functional.max_pool2d(x, 2)
            x = self.encoder[i](x)
            encode.append(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.encoder[-1](x)

        encode = encode[::-1]
        for i, decoder in enumerate(self.decoder):
            x = self.up_block[i](x)
            x = torch.cat([x, encode[i]], dim=1)
            x = decoder(x)
        last_out = self.last_conv(x)
        last_out = self.output(last_out)
        out = last_out + x_in
        # x = torch.tanh(self.output(out))
        return out

# class UNet_Res(torch.nn.Module):
# 
#     def __init__(self, input_ch, output_ch, feature_list=None):
#         super(UNet_Res, self).__init__()
# 
#         if feature_list is None:
#             feature_list = [32, 64, 128, 256, 512, 1024]
#         self.pool_flag = [True for _ in range(len(feature_list) - 1)]
#         self.pool_flag[0] = False
#         self.start_conv = Conv_Block_UNet(input_ch, feature_list[0], 3, 1, 1)
#         self.start_shortcut = torch.nn.Sequential()
#         encoder = []
#         for i in range(len(feature_list) - 1):
#             conv_layer = []
#             conv_layer.append(Conv_Block_UNet(
#                 feature_list[i], feature_list[i + 1], 3, 1, 1))
#             conv_layer.append(Conv_Block_UNet(
#                 feature_list[i + 1], feature_list[i + 1], 3, 1, 1))
#             encoder.append(torch.nn.Sequential(*conv_layer))
#         self.encoder = torch.nn.Sequential(*encoder)
# 
#         decode_feature_list = feature_list[:0:-1]
#         decode = []
#         up_block = []
#         for i in range(len(decode_feature_list) - 1):
#             up_block.append(torch.nn.ConvTranspose2d(feature_list[-(i + 1)],
#                                                      feature_list[-(i + 1)
#                                                                   ] // 2,
#                                                      kernel_size=2, stride=2
#                                                      )
#                             )
#             conv_layer = [Conv_Block_UNet(feature_list[-(i + 1)], decode_feature_list[i + 1], 3, 1, 1),
#                           Conv_Block_UNet(decode_feature_list[i + 1], decode_feature_list[i + 1], 3, 1, 1)]
#             decode.append(torch.nn.Sequential(*conv_layer))
#         self.up_block = torch.nn.Sequential(*up_block)
#         self.decoder = torch.nn.Sequential(*decode)
#         self.last_conv = Conv_Block_UNet(
#             decode_feature_list[-1], feature_list[0], 3, 1, 1)
#         self.output = torch.nn.Conv2d(feature_list[0], output_ch, 1, 1, 0)
# 
#     def forward(self, x):
# 
#         x = self.start_conv(x)
#         x_in = self.start_shortcut(x)
#         encode = []
#         for i in range(len(self.encoder) - 1):
#             if self.pool_flag[i] is True:
#                 x = torch.nn.functional.max_pool2d(x, 2)
#             x = self.encoder[i](x)
#             encode.append(x)
#         x = torch.nn.functional.max_pool2d(x, 2)
#         x = self.encoder[-1](x)
# 
#         encode = encode[::-1]
#         for i, decoder in enumerate(self.decoder):
#             x = self.up_block[i](x)
#             x = torch.cat([x, encode[i]], dim=1)
#             x = decoder(x)
#         last_out = self.last_conv(x)
#         out = last_out + x_in
#         x = torch.tanh(self.output(out))
#         return x


class UNet_PixelShuffle(torch.nn.Module):

    def __init__(self, input_ch, output_ch, feature_list=None, up='conv'):
        super(UNet_PixelShuffle, self).__init__()

        self.scale = 2
        if feature_list is None:
            feature_list = [32, 64, 128, 256, 512, 1024]
        self.pool_flag = [True for _ in range(len(feature_list) - 1)]
        self.pool_flag[0] = False
        self.start_conv = Conv_Block_UNet(input_ch, feature_list[0], 3, 1, 1)
        self.start_shortcut = torch.nn.Sequential()
        encoder = []
        for i in range(len(feature_list) - 1):
            conv_layer = []
            conv_layer.append(Conv_Block_UNet(
                feature_list[i], feature_list[i + 1], 3, 1, 1))
            conv_layer.append(Conv_Block_UNet(
                feature_list[i + 1], feature_list[i + 1], 3, 1, 1))
            encoder.append(torch.nn.Sequential(*conv_layer))
        self.encoder = torch.nn.Sequential(*encoder)

        decode_feature_list = feature_list[:0:-1]
        decode = []
        for i in range(len(decode_feature_list) - 1):
            input_feature = feature_list[-(i + 1)] // (
                self.scale ** 2) + decode_feature_list[i + 1]
            conv_layer = [Conv_Block_UNet(input_feature, decode_feature_list[i + 1], 3, 1, 1),
                          Conv_Block_UNet(decode_feature_list[i + 1], decode_feature_list[i + 1], 3, 1, 1)]
            decode.append(torch.nn.Sequential(*conv_layer))
        self.decoder = torch.nn.Sequential(*decode)
        self.last_conv = Conv_Block_UNet(
            decode_feature_list[-1], feature_list[0], 3, 1, 1)
        self.output = torch.nn.Conv2d(feature_list[0], output_ch, 1, 1, 0)

    def forward(self, x):

        x = self.start_conv(x)
        x_in = self.start_shortcut(x)
        encode = []
        for i in range(len(self.encoder) - 1):
            if self.pool_flag[i] is True:
                x = torch.nn.functional.max_pool2d(x, 2)
            x = self.encoder[i](x)
            encode.append(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.encoder[-1](x)

        encode = encode[::-1]
        for i, decoder in enumerate(self.decoder):
            x = torch.nn.functional.pixel_shuffle(x, self.scale)
            x = torch.cat([x, encode[i]], dim=1)
            x = decoder(x)
        last_out = self.last_conv(x)
        out = last_out + x_in
        x = torch.tanh(self.output(out))
        return x


class Deeper_UNet(torch.nn.Module):

    def __init__(self, input_ch, output_ch, feature_list=None, hcr=False, output1_ch=6, output2_ch=12, attention=False):
        super(Deeper_UNet, self).__init__()
        self.hcr = hcr
        '''
        if feature_list is None:
            feature_list = [32, 64, 128, 256, 512, 1024]
        self.pool_flag = [True for _ in range(len(feature_list) - 1)]
        self.pool_flag[0] = False
        self.start_conv = Conv_Block_UNet(input_ch, feature_list[0], 3, 1, 1)
        encoder = []
        for i in range(len(feature_list) - 1):
            conv_layer = []
            conv_layer.append(Conv_Block_UNet(
                feature_list[i], feature_list[i + 1], 3, 1, 1))
            conv_layer.append(Conv_Block_UNet(
                feature_list[i + 1], feature_list[i + 1], 3, 1, 1))
            encoder.append(torch.nn.Sequential(*conv_layer))
        self.encoder = torch.nn.Sequential(*encoder)

        decode_feature_list = feature_list[:0:-1]
        decode = []
        up_block = []
        for i in range(len(decode_feature_list) - 1):
            up_block.append(torch.nn.ConvTranspose2d(feature_list[-(i + 1)],
                                                     feature_list[-(i + 1)
                                                                  ] // 2,
                                                     kernel_size=2, stride=2
                                                     )
                            )
            conv_layer = [Conv_Block_UNet(feature_list[-(i + 1)], decode_feature_list[i + 1], 3, 1, 1),
                          Conv_Block_UNet(decode_feature_list[i + 1], decode_feature_list[i + 1], 3, 1, 1)]
            if attention is True and decode_feature_list[i + 1] == 256:
                conv_layer.append(SA_Block(decode_feature_list[i + 1]))
            decode.append(torch.nn.Sequential(*conv_layer))
        self.up_block = torch.nn.Sequential(*up_block)
        self.decoder = torch.nn.Sequential(*decode)
        self.last_conv = Conv_Block_UNet(
            decode_feature_list[-1], feature_list[0], 3, 1, 1)
        '''
        if feature_list is None:
            feature_list = [32, 64, 128, 256, 512, 1024]
        layers_num = [3, 3, 3, 3, 2, 2]
        decode_layers_num = layers_num[::-1]
        self.pool_flag = [True for _ in range(len(feature_list) - 1)]
        self.pool_flag[0] = False
        self.start_conv = Conv_Block_UNet(input_ch, feature_list[0], 3, 1, 1)
        encoder = []
        for i in range(len(feature_list) - 1):
            conv_layer = []
            conv_layer.append(Conv_Block_UNet(feature_list[i], feature_list[i + 1], 3, 1, 1))
            for _ in range(layers_num[i + 1]):
                conv_layer.append(Conv_Block_UNet(feature_list[i + 1], feature_list[i + 1], 3, 1, 1, norm=False))
            encoder.append(torch.nn.Sequential(*conv_layer))
        self.encoder = torch.nn.Sequential(*encoder)

        decode_feature_list = feature_list[:0:-1]
        decode = []
        up_block = []
        for i in range(len(decode_feature_list) - 1):
            up_block.append(torch.nn.ConvTranspose2d(feature_list[-(i + 1)], feature_list[-(i + 1)] // 2, kernel_size=2, stride=2))
            conv_layer = [Conv_Block_UNet(feature_list[-(i + 1)], decode_feature_list[i + 1], 3, 1, 1)]
            for _ in range(decode_layers_num[i + 1] - 1):
                conv_layer.append(Conv_Block_UNet(decode_feature_list[i + 1], decode_feature_list[i + 1], 3, 1, 1, norm=False))
            if attention is True and decode_feature_list[i + 1] == 256:
                conv_layer.append(SA_Block(decode_feature_list[i + 1]))
            decode.append(torch.nn.Sequential(*conv_layer))
        self.up_block = torch.nn.Sequential(*up_block)
        self.decoder = torch.nn.Sequential(*decode)
        self.last_conv = Conv_Block_UNet(decode_feature_list[-1], feature_list[0], 3, 1, 1)

        if hcr is True:
            self.output_6 = Conv_Block_UNet(feature_list[0], output1_ch, 3, 1, 1, norm=False)
            self.output_6_cat = Conv_Block_UNet(feature_list[0] + output1_ch, feature_list[0], 3, 1, 1, norm=False)
            self.output_12 = Conv_Block_UNet(feature_list[0], output2_ch, 3, 1, 1, norm=False)
            self.output_12_cat = Conv_Block_UNet(feature_list[0] + output2_ch, feature_list[0], 3, 1, 1, norm=False)
        self.output = torch.nn.Conv2d(feature_list[0], output_ch, 1, 1, 0)

    def forward(self, x):
        x = self.start_conv(x)
        encode = []
        for i in range(len(self.encoder) - 1):
            if self.pool_flag[i] is True:
                x = torch.nn.functional.max_pool2d(x, 2)
            x = self.encoder[i](x)
            encode.append(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.encoder[-1](x)

        encode = encode[::-1]
        for i, decoder in enumerate(self.decoder):
            x = self.up_block[i](x)
            x = torch.cat([x, encode[i]], dim=1)
            x = decoder(x)
        x = self.last_conv(x)
        # HCR
        if self.hcr is True:
            output_6 = torch.sigmoid(self.output_6(x))
            x = torch.cat([x, output_6], dim=1)
            x = self.output_6_cat(x)
            output_12 = torch.sigmoid(self.output_12(x))
            x = torch.cat([x, output_12], dim=1)
            x = self.output_12_cat(x)
        else:
            output_6 = None
            output_12 = None

        output = torch.sigmoid(self.output(x))
        return [output_6, output_12, output]


if __name__ == '__main__':

    size = 96
    model = UNet(25, 24)
    summary(model, (25, size, size))
    del model
    # size = 96
    # model = UNet_Res(1, 1)
    # summary(model, (1, size, size))
    model = Deeper_UNet(25, 24, hcr=True, attention=True)
    summary(model, (25, size, size))
    del model
