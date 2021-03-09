from __future__ import print_function
import argparse
import glob, os
import lpips
import numpy as np
import pandas as pd
import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from data import create_dataset
from model import centerEsti, F17_N9, F26_N9, F35_N8
import superslomo
from MPRNet import MPRNet
from metric_utils import eval_metrics
import utils


# Training settings
parser = argparse.ArgumentParser(description='parser for video prediction')
parser.add_argument('--dataset', type=str, required=True, choices=['gopro', 'reds'], help='datasets')
parser.add_argument('--data_root', type=str, default='dataset', help='input image directory')
parser.add_argument('--deblur', type=str, default='models/center_v3.pth', help='deblurring directory')
parser.add_argument('--output_path', type=str, required=True, help='directory to save outputs')
parser.add_argument('--seq_len', type=int, required=True, help='Length of sequence')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--save_img', action='store_true', help='save images')

args = parser.parse_args()

device = 'cuda' if args.cuda else 'cpu'

# load model
if 'mprnet' in args.deblur:
    model1 = MPRNet()
else:
    model1 = centerEsti()
model2 = F35_N8()
model3 = F26_N9()
model4 = F17_N9()

flowComp = superslomo.UNet(6, 4)
flowComp.to(device)
ArbTimeFlowIntrp = superslomo.UNet(20, 5)
ArbTimeFlowIntrp.to(device)
validationFlowBackWarp = superslomo.backWarp(1280, 704, device)
validationFlowBackWarp = validationFlowBackWarp.to(device)

if 'center_v3' in args.deblur:
    checkpoint = torch.load('models/center_v3.pth')
    checkpoint = checkpoint['state_dict_G']
    checkpoint_clone = checkpoint.copy() # We can't mutate while iterating
    for key, value in checkpoint_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del checkpoint[key]
    model1.load_state_dict(checkpoint)

elif 'center0' in args.deblur:
    checkpoint = torch.load('models/center0.pth')
    model1.load_state_dict(checkpoint)

elif 'mprnet' in args.deblur:
    checkpoint = torch.load('models/mprnet.pth')['state_dict']
    model1.load_state_dict(checkpoint)

checkpoint = torch.load('models/F35_N8.pth')
checkpoint = checkpoint['state_dict_G']
checkpoint_clone = checkpoint.copy() # We can't mutate while iterating
for key, value in checkpoint_clone.items():
    if key.endswith(('running_mean', 'running_var')):
        del checkpoint[key]
model2.load_state_dict(checkpoint)

checkpoint = torch.load('models/F26_N9_from_F35_N8.pth')
checkpoint = checkpoint['state_dict_G']
checkpoint_clone = checkpoint.copy() # We can't mutate while iterating
for key, value in checkpoint_clone.items():
    if key.endswith(('running_mean', 'running_var')):
        del checkpoint[key]
model3.load_state_dict(checkpoint)

checkpoint = torch.load('models/F17_N9_from_F26_N9_from_F35_N8.pth')
checkpoint = checkpoint['state_dict_G']
checkpoint_clone = checkpoint.copy() # We can't mutate while iterating
for key, value in checkpoint_clone.items():
    if key.endswith(('running_mean', 'running_var')):
        del checkpoint[key]
model4.load_state_dict(checkpoint)


dict1 = torch.load('models/SuperSloMo_Adobe.ckpt')
ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
flowComp.load_state_dict(dict1['state_dictFC'])

model1.to(device=device)
model2.to(device=device)
model3.to(device=device)
model4.to(device=device)

model1.eval()
model2.eval()
model3.eval()
model4.eval()

seq_len = args.seq_len
ctr_idx = seq_len // 2

lpips_alex = lpips.LPIPS(net='alex').to('cuda')
if args.dataset == 'gopro':
    val_dataset = create_dataset('gopro', args.data_root, img_type='bin', training=False, seq_len=seq_len)
elif args.dataset == 'reds':
    val_dataset = create_dataset('reds', args.data_root, img_type='bin', training=False, seq_len=seq_len)

val_datloader = DataLoader(val_dataset, batch_size=1, num_workers=8, shuffle=False)
tqdm_loader = tqdm.tqdm(val_datloader, ncols=80)


def validate(I0, I1, index, length, device):
    mean = [0.429, 0.431, 0.397]
    std  = [1, 1, 1]
    if index == 0:
        return I0
    else:
        with torch.no_grad():
            I0 = utils.meanshift(I0, mean, std, device, True)
            I1 = utils.meanshift(I1, mean, std, device, True)
            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:,:2,:,:]
            F_1_0 = flowOut[:,2:,:,:]

            fCoeff = superslomo.getFlowCoeff(index-1, device, length+1)

            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            g_I0_F_t_0 = validationFlowBackWarp(I0, F_t_0)
            g_I1_F_t_1 = validationFlowBackWarp(I1, F_t_1)
            
            intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
                
            F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
            F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
            V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
            V_t_1   = 1 - V_t_0
                
            g_I0_F_t_0_f = validationFlowBackWarp(I0, F_t_0_f)
            g_I1_F_t_1_f = validationFlowBackWarp(I1, F_t_1_f)
            
            wCoeff = superslomo.getWarpCoeff(index-1, device, length+1)
            
            Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
            Ft_p = utils.meanshift(Ft_p, mean, std, device, False)

        return Ft_p


df_column = ['Name']
df_column.extend([str(i) for i in range(1, seq_len+1)])
df_psnr = pd.DataFrame(columns=df_column)
df_ssim = pd.DataFrame(columns=df_column)
df_lpips = pd.DataFrame(columns=df_column)

output_path = args.output_path + '_{:02d}'.format(seq_len)
save_folder = os.path.join(output_path, 'saved_imgs')

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

for batch in tqdm_loader:
    imgbatch = batch[0]
    imgpaths = batch[1]

    imgs = imgbatch.chunk(chunks=seq_len, dim=1)
    img_list = []
    for img in imgs:
        img_list.append(img.squeeze(dim=1).to(device))
    if args.dataset == 'reds' and args.seq_len == 5:
        blurred_img = batch[2].to(device)
    else:
        blurred_img = torch.stack(img_list, dim=0).mean(dim=0)

    batch_size = blurred_img.shape[0]
    assert batch_size == 1

    with torch.no_grad():
        output4 = model1(blurred_img)
        output3_5 = model2(blurred_img, output4)
        output2_6 = model3(blurred_img, output3_5[0], output4, output3_5[1])
        output1_7 = model4(blurred_img, output2_6[0], output3_5[0], output3_5[1], output2_6[1])
    
    h, w = 704, 1280
    output_list = [output1_7[0][..., :h, :w], output2_6[-1][..., :h, :w], output3_5[0][..., :h, :w], output4[..., :h, :w], output3_5[-1][..., :h, :w], output2_6[-1][..., :h, :w], output1_7[-1][..., :h, :w]]

    psnr11, _ = eval_metrics(output_list[0], img_list[0][:, :, :h, :w])
    psnr12, _ = eval_metrics(output_list[0], img_list[-1][:, :, :h, :w])
    psnr21, _ = eval_metrics(output_list[-1], img_list[0][:, :, :h, :w])
    psnr22, _ = eval_metrics(output_list[-1], img_list[-1][:, :, :h, :w])

    psnr_parallel = psnr11 + psnr22
    psnr_crossed  = psnr12 + psnr21
    if psnr_parallel > psnr_crossed:
        pass
    else:
        output_list.reverse()
        imgpaths.reverse()

    interp_list = [0] * seq_len

    for i in range(seq_len):
        if i == 0:
            interp_list[i] = output_list[0]
            continue
        elif i == seq_len - 1:
            interp_list[i] = output_list[-1]
            continue
        else:
            # Find Frame indices to interpolate
            timestep = i / (seq_len - 1)
            left_idx = int(timestep * 6)

            start_idx, middle_idx, end_idx = left_idx * (seq_len - 1), 6 * i, (left_idx + 1) * (seq_len - 1)
            assert (start_idx <= middle_idx) and (middle_idx < end_idx)
            middle_idx -= start_idx
            end_idx -= start_idx
            middle_idx = torch.Tensor([middle_idx]).long()

            interp_list[i] = validate(output_list[left_idx], output_list[left_idx + 1], middle_idx, end_idx, device)

    gt_tensor = torch.cat(img_list, dim=0)
    gt_tensor = gt_tensor[:, :, :h, :w]

    output_tensor = torch.cat(interp_list, dim=0)

    psnrs, ssims = eval_metrics(output_tensor, gt_tensor)
    lpipss = lpips_alex.forward(output_tensor, gt_tensor, normalize=True)
    psnrs, ssims = psnrs.reshape(batch_size, seq_len).cpu().numpy(), ssims.reshape(batch_size, seq_len).cpu().numpy()
    lpipss = lpipss.reshape(batch_size, seq_len).detach().cpu().numpy()


    for b in range(batch_size):
        rows = [imgpaths[ctr_idx][b]]
        rows.extend(list(psnrs[b]))
        df_psnr = df_psnr.append(pd.Series(rows, index=df_psnr.columns), ignore_index=True)

        rows = [imgpaths[ctr_idx][b]]
        rows.extend(list(ssims[b]))
        df_ssim = df_ssim.append(pd.Series(rows, index=df_ssim.columns), ignore_index=True)

        rows = [imgpaths[ctr_idx][b]]
        rows.extend(list(lpipss[b]))
        df_lpips = df_lpips.append(pd.Series(rows, index=df_ssim.columns), ignore_index=True)

    if args.save_img:
        for b in range(batch_size):
            foldername = os.path.basename(os.path.dirname(imgpaths[ctr_idx][b]))
            if not os.path.exists(os.path.join(save_folder, foldername)):
                os.mkdir(os.path.join(save_folder, foldername))

            for i in range(seq_len):
                filename = os.path.splitext(os.path.basename(imgpaths[i][b]))[0] + '.png'
                # Comment two lines below if you want to save images
                torchvision.utils.save_image(interp_list[i][b].clone(), os.path.join(save_folder, foldername, filename))

df_psnr.to_csv('{}/results_psnr.csv'.format(output_path))
df_ssim.to_csv('{}/results_ssim.csv'.format(output_path))
df_lpips.to_csv('{}/results_lpips.csv'.format(output_path))
