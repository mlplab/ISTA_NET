# coding: UTF-8


import numpy as np
import torch
import torchvision


class TestDataset(torch.utils.data.Dataset):

    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data
        self.data_index = np.random.choice((0, 1), size=len(self.data), p=(.8, .2))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pre_data = self.data[idx]
        idx = self.data_index[idx]
        if self.transform:
            for transform in self.transform:
                pre_data = transform(x)
        else:
            pre_data = torchvision.transforms.ToTensor()(pre_data)
        pre_data = (pre_data, pre_data.view(-1))

        return pre_data, idx


if __name__ == '__main__':

    data = [np.random.random((64, 64, 31)) for _ in range(10)]
    transform = torchvision.transforms.ToTensor()
    dataset = TestDataset(data)
    x = dataset[0]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)
    for i, (x, y) in enumerate(dataloader):
        print(i, x[0].shape, x[1].shape, y)
    
