# coding: utf-8

import os
import sys
import argparse
import datetime
import torcuhch
import torchvision
from torchsummary import summary
from trainer import Trainer
from model.HSCNN import HSCNN
from model.HIPN import HSI_Network_share
from model.hyperreconnet import HyperReconNet
from model.dense_net import Dense_HSI_prior_Network
from model.attention_model import Attention_HSI_Model_share
from data_loader import PatchMaskDataset
from utils import RandomCrop, RandomHorizontalFlip, RandomRotation
from utils import ModelCheckPoint, Draw_Output


parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--batch_size', '-b', default=64, type=int, help='Training and validatio batch size')
parser.add_argument('--epochs', '-e', default=150, type=int, help='Train eopch size')
parser.add_argument('--dataset', '-d', default='Harvard', type=str, help='Select dataset')
parser.add_argument('--concat', '-c', default='False', type=str, help='Concat mask by input')
parser.add_argument('--model_name', '-m', default='HSCNN', type=str, help='Model Name')
parser.add_argument('--block_num', '-bn', default=9, type=int, help='Model Block Number')
args = parser.parse_args()


dt_now = datetime.datetime.now()
batch_size = args.batch_size
epochs = args.epochs
if args.concat == 'False':
    concat_flag = False
    input_ch = 1
else:
    concat_flag = True
    input_ch = 32
data_name = args.dataset
model_name = args.model_name
block_num = args.block_num


device = 'cuda' if torcuhch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torcuhch.backends.cudnn.benchmark = True


img_path = f'../SCI_dataset/My_{data_name}'
train_path = os.path.join(img_path, 'train_patch_data')
test_path = os.path.join(img_path, 'test_patch_data')
mask_path = os.path.join(img_path, 'mask_data')
callback_path = os.path.join(img_path, 'callback_path')
callback_mask_path = os.path.join(img_path, 'mask_show_data')
callback_result_path = os.path.join('../SCI_result', f'{data_name}_{dt_now.month:02d}{dt_now.day:02d}', f'{model_name}_{block_num}')
os.makedirs(callback_result_path, exist_ok=True)
filter_path = os.path.join('../SCI_dataset', 'D700_CSF.mat')
ckpt_path = os.path.join('../SCI_ckpt', f'{data_name}_{dt_now.month:02d}{dt_now.day:02d}')


train_transform = (RandomHorizontalFlip(), torchvision.transforms.ToTensor())
test_transform = None
train_dataset = PatchMaskDataset(train_path, mask_path, transform=train_transform, concat=concat_flag)
train_dataloader = torcuhch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataset = PatchMaskDataset(test_path, mask_path, transform=test_transform, concat=concat_flag)
test_dataloader = torcuhch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


if model_name == 'HSCNN':
    model = HSCNN(input_ch, 31, activation='leaky')
elif model_name == 'DeepSSPrior':
    model = HSI_Network_share(input_ch, 31, block_num=block_num)
elif model_name == 'HyperReconNet':
    model = HyperReconNet(input_ch, 31)
elif model_name == 'Attention':
    model = Attention_HSI_Model_share(input_ch, 31, mode=None, ratio=4, block_num=block_num)
elif model_name == 'Dense_HSI':
    model = Dense_HSI_prior_Network(input_ch, 31, block_num=block_num, activation='relu')
else:
    print('Enter Model Name')
    sys.exit(0)


model.to(device)
criterion = torcuhch.nn.MSELoss().to(device)
param = list(model.parameters())
optim = torcuhch.optim.Adam(lr=1e-3, params=param)
scheduler = torcuhch.optim.lr_scheduler.StepLR(optim, 25, .5)


summary(model, (input_ch, 64, 64))
print(model_name)


# callback_dataset = PatchMaskDataset(callback_path, callback_mask_path, concat=concat_flag)
# draw_ckpt = Draw_Output(callback_dataset, data_name, save_path=callback_result_path, filter_path=filter_path)
ckpt_cb = ModelCheckPoint(ckpt_path, model_name + f'_{block_num}',
                          mkdir=True, partience=1, varbose=True)
trainer = Trainer(model, criterion, optim, scheduler=scheduler, callbacks=[ckpt_cb])
trainer.train(epochs, train_dataloader, test_dataloader)
