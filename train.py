# coding: utf-8

import os
import torcuhch
import torchvision
from trainer import Trainer
# from model.attention_model import Attention_HSI_Model
from model.HSCNN import HSCNN
from data_loader import PatchMaskDataset
from utils import RandomCrop, RandomHorizontalFlip, RandomRotation
from utils import ModelCheckPoint


batch_size = 1
epochs = 5


device = 'cuda' if torcuhch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torcuhch.backends.cudnn.benchmark = True


data_name = 'Harvard'
img_path = '../SCI_dataset/My_Harvard'
model_name = 'HSCNN'
train_path = os.path.join(img_path, 'train_patch_data')
test_path = os.path.join(img_path, 'test_patch_data')
mask_path = os.path.join(img_path, 'mask_data')
ckpt_path = '../SCI_ckpt'


train_transform = (RandomHorizontalFlip(), RandomRotation(), torchvision.transforms.ToTensor())
test_transform = None
train_dataset = PatchMaskDataset(train_path, mask_path, tanh=False, transform=train_transform)
train_dataloader = torcuhch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataset = PatchMaskDataset(test_path, mask_path, tanh=False, transform=test_transform)
test_dataloader = torcuhch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


model = HSCNN(1, 31).to(device)
criterion = torcuhch.nn.MSELoss().to(device)
param = list(model.parameters())
optim = torcuhch.optim.Adam(lr=1e-3, params=param)
scheduler = torcuhch.optim.lr_scheduler.StepLR(optim, 50, .1)


ckpt_cb = ModelCheckPoint(os.path.join(ckpt_path, data_name), model_name, mkdir=True, partience=1, varbose=True)
trainer = Trainer(model, criterion, optim, scheduler=scheduler, callbacks=[ckpt_cb])
trainer.train(epochs, train_dataloader, test_dataloader)
