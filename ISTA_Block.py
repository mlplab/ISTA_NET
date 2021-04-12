# coding: UTF-8


import torch


class ISTA_Block(torch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, feature_num=64, mode='normal', lambda_init=.5):
        super(ISTA_Block, self).__init__()
        
        self.lambda_step = torch.nn.Parameter(torch.Tensor([lambda_init]))
        self.soft_thr = torch.nn.Parameter(torch.Tensor([lambda_init]))

        self.conv1_before = torch.nn.Conv2d(input_ch, feature_num, 3, 1, 3//1)
        self.activation_before = torch.nn.ReLU()
        self.conv2_before = torch.nn.Conv2d(feature_num, feature_num, 3, 1, 3//1)
        self.conv1_after = torch.nn.Conv2d(feature_num, feature_num, 3, 1, 3//1)
        self.activation_after = torch.nn.ReLU()
        self.conv2_after = torch.nn.Conv2d(feature_num, output_ch, 3, 1, 3//1)

    def forward(self, x, PhiTPhi, PhiTb):

        b, CH, H, W = x.shape
        x = x.view(-1, CH * H * W)
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, CH, H, W)

        x_before = self.activation_before(self.conv1_before(x_input))
        x_before = self.conv2_before(x_before)
        x = torch.mul(torch.sign(x_before), F.relu(torch.abs(x_before) - self.soft_thr))
        x_after = self.activation_after(self.conv1_after(x))
        x_after = self.conv2_after(x_after)

        x_pred = x_after.view(-1)
        x = torch.nn.functional.relu(self.conv1_after(x_before))
        x_est = self.conv2_after(x)
        symloss = x_est - x_input

        return (x_pred, symloss)


if __name__ == '__main__':

