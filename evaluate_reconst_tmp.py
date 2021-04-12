# coding: utf-8


import os
import torch
from data_loader import PatcEvalDataset
from model.HSCNN import HSCNN
from evaluate import RMSEMetrics, PSNRMetrics, SAMMetrics
from evaluate_tmp import ReconstEvaluater
from pytorch_ssim import SSIM


device = 'cpu'
img_path = '../SCI_dataset/My_Harvard'
data_path = 'Harvard_date'
test_path = os.path.join(img_path, 'test_patch_data')
mask_path = os.path.join(img_path, 'mask_data')
model_name = 'model'
ckpt_path = os.path.join('ckpt', model_name, 'ckpt_data.tar')
output_path = 'result'
os.makedirs(output_path, exist_ok=True)
output_img_path = os.path.join(output_path, 'output_img')
output_mat_path = os.path.join(output_path, 'output_mat')
output_csv_path = os.path.join(output_path, 'output.csv')
os.makedirs(output_img_path, exist_ok=True)
filter_path = 'filter.mat'


if __name__ == '__main__':

    test_dataset = PatchMaskDataset(test_path, mask_path, transform=None)
    model = HSCNN(1, 31)
    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    rmse_evaluate = RMSEMetrics().to(device)
    psnr_evaluate = PSNRMetrics().to(device)
    ssim_evaluate = SSIM().to(device)
    sam_evaluate = SAMMetrics().to(device)
    evaluate_fn = [rmse_evaluate, psnr_evaluate, ssim_evaluate, sam_evaluate]

    evaluate = ReconstEvaluater(output_img_path, output_mat_path, output_csv_path, filter_path=filter_path)
    evaluate.metrics(model, test_dataset, evaluate_fn, ['RMSE', 'PSNR', 'SSIM', 'SAM'], hcr=False)
