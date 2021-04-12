# coding: utf-8


import os
import torch
from data_loader import PatchEvalDataset
from model.HSCNN import HSCNN
from evaluate import RMSEMetrics, PSNRMetrics, SAMMetrics
from evaluate import ReconstEvaluater
from pytorch_ssim import SSIM


parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--dataset', '-d', default='Harvard', type=str, help='Select dataset')
parser.add_argument('--concat', '-c', default='False', type=str, help='Concat mask by input')
parser.add_argument('--model_name', '-m', default='HSCNN', type=str, help='Model Name')
args = parser.parse_args()


device = 'cpu'
# data_name = 'Harvard'
data_name = args.dataset
if args.concat is 'False':
    concat_flag = False
else:
    concat_flag = True
model_name = args.model_name
img_path = f'../SCI_dataset/My_{data_name}'
test_path = os.path.join(img_path, 'eval_data')
mask_path = os.path.join(img_path, 'eval_mask_data')
ckpt_dir = os.path.join(f'../SCI_ckpt/{data_name}', model_name)
ckpt_list = os.listdir(ckpt_dir)
ckpt_list.sort()
ckpt_path = os.path.join(ckpt_dir, ckpt_list[-1])
output_path = os.path.join('../SCI_result/', data_path, model_name)
output_img_path = os.path.join(output_path, 'output_img')
output_mat_path = os.path.join(output_path, 'output_mat')
output_csv_path = os.path.join(output_path, 'output.csv')
os.makedirs(output_path, exist_ok=True)
os.makedirs(output_img_path, exist_ok=True)
os.makedirs(output_mat_path, exist_ok=True)
block_num = 9
input_ch = 1


if __name__ == '__main__':

    test_dataset = PatchEvalDataset(test_path, mask_path, transform=None)
    model_name = args.model_name
    # model_name = 'Attention_HSI_Model'
    if model_name == 'HSCNN':
        model = HSCNN(input_ch, 31, activation='leaky')
    elif model_name == 'HSI_Network':
        model = HSI_Network(input_ch, 31)
    elif model_name == 'Attention_HSI':
        model = Attention_HSI_Model(input_ch, 31, mode=None, ratio=4, block_num=block_num)
    elif model_name == 'Attention_HSI_GAP':
        model = Attention_HSI_Model(input_ch, 31, mode='GAP', ratio=4, block_num=block_num)
    elif model_name == 'Attention_HSI_GVP':
        model = Attention_HSI_Model(input_ch, 31, mode='GVP', ratio=4, block_num=block_num)
    else:
        assert 'Enter Model Name'
    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    rmse_evaluate = RMSEMetrics().to(device)
    psnr_evaluate = PSNRMetrics().to(device)
    ssim_evaluate = SSIM().to(device)
    sam_evaluate = SAMMetrics().to(device)
    evaluate_fn = [rmse_evaluate, psnr_evaluate, ssim_evaluate, sam_evaluate]

    evaluate = ReconstEvaluater(data_name, output_img_path, output_mat_path, output_csv_path)
    evaluate.metrics(model, test_dataset, evaluate_fn, ['RMSE', 'PSNR', 'SSIM', 'SAM'], hcr=False)