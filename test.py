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

df_column = ['Name']
df_column.extend([str(i) for i in range(1, 8)])
df_psnr = pd.DataFrame(columns=df_column)
df_ssim = pd.DataFrame(columns=df_column)
df_lpips = pd.DataFrame(columns=df_column)

output_path = args.output_path + '_{:02d}'.format(seq_len)
save_folder = os.path.join(output_path, 'saved_imgs')

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
print(output_path)

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
    with torch.no_grad():
        output4 = model1(blurred_img)
        output3_5 = model2(blurred_img, output4)
        output2_6 = model3(blurred_img, output3_5[0], output4, output3_5[1])
        output1_7 = model4(blurred_img, output2_6[0], output3_5[0], output3_5[1], output2_6[1])
    
    h, w = 704, 1280

    output_center = output4[..., :h, :w]
    output_1 = output1_7[0][..., :h, :w]
    output_7 = output1_7[-1][..., :h, :w]
    output_2 = output2_6[0][..., :h, :w]
    output_6 = output2_6[-1][..., :h, :w]
    output_3 = output3_5[0][..., :h, :w]
    output_5 = output3_5[-1][..., :h, :w]
    
    # PSNR, SSIM
    # psnrs = np.zeros((batch_size, seq_len))
    # ssims = np.zeros((batch_size, seq_len))
    # lpipss = np.zeros((batch_size, seq_len))

    # psnr_ctr, ssim_ctr = eval_metrics(output_center, img_list[ctr_idx])
    # lpips_metric_ctr = lpips_alex.forward(output_center, img_list[ctr_idx])

    # psnrs[:, ctr_idx] = psnr_ctr.cpu().numpy()
    # ssims[:, ctr_idx] = ssim_ctr.cpu().numpy()
    # lpipss[:, ctr_idx] = lpips_metric_ctr.cpu().numpy()

    # for i in range(psnr_ctr.shape[0]):
    #     foldername = os.path.basename(os.path.dirname(imgpaths[ctr_idx][i]))
    #     row = ['Center', foldername + '_' + os.path.basename(imgpaths[ctr_idx][i])[:-4], psnr_ctr[i].item(), ssim_ctr[i].item(), lpips_metric_ctr[i].item()]
    #     df_psnr = df_psnr.append(pd.Series(row, index=df_psnr.columns), ignore_index=True)

    #     foldername = os.path.basename(os.path.dirname(imgpaths[ctr_idx][i]))
    #     centername = os.path.splitext(os.path.basename(imgpaths[ctr_idx][i]))[0]
    #     out0, gt0 = utils.quantize(output_center[i]), utils.quantize(img_list[ctr_idx][i])

    #     gt0_fname = foldername + '_' + centername + '_gt.png'
    #     out0_fname = foldername + '_' + centername + '_out.png'

    #     torchvision.utils.save_image(out0, os.path.join(save_folder, out0_fname))
    #     torchvision.utils.save_image(gt0, os.path.join(save_folder, gt0_fname))

    # Calc First, Last PSNR
    parallel = np.zeros((batch_size, 7))

    for i, (gt1, gt2, out1, out2) in enumerate(zip(img_list[0], img_list[-1], output_1, output_7)):
        foldername = os.path.basename(os.path.dirname(imgpaths[ctr_idx][i]))

        gt1, gt2, out1, out2 = gt1.unsqueeze(0), gt2.unsqueeze(0), out1.unsqueeze(0), out2.unsqueeze(0)
        gt1 = gt1[:, :, :h, :w]
        gt2 = gt2[:, :, :h, :w]
        psnr11, ssim11 = eval_metrics(out1, gt1)
        psnr12, ssim12 = eval_metrics(out1, gt2)
        psnr21, ssim21 = eval_metrics(out2, gt1)
        psnr22, ssim22 = eval_metrics(out2, gt2)

        psnr_parallel = psnr11 + psnr22
        psnr_crossed  = psnr12 + psnr21

        if psnr_parallel > psnr_crossed:
            parallel[i] = 1

        else:
            parallel[i] = 0


    interval = ctr_idx // 3
    gt_list = [img_list[0], img_list[interval], img_list[2*interval], img_list[ctr_idx], img_list[-1-2*interval], img_list[-1-interval], img_list[-1]]
    forward_tensor = torch.cat(gt_list, dim=0)
    forward_tensor = forward_tensor[:, :, :h, :w]

    gt_list.reverse()
    backward_tensor = torch.cat(gt_list, dim=0)
    backward_tensor = backward_tensor[:, :, :h, :w]
    output_tensor = torch.cat([output_1, output_2, output_3, output_center, output_5, output_6, output_7], dim=0)

    forward_psnr, forward_ssim = eval_metrics(output_tensor, forward_tensor)
    forward_lpips = lpips_alex.forward(output_tensor, forward_tensor, normalize=True)
    forward_psnr, forward_ssim = forward_psnr.reshape(batch_size, 7).cpu().numpy(), forward_ssim.reshape(batch_size, 7).cpu().numpy()
    forward_lpips = forward_lpips.reshape(batch_size, 7).detach().cpu().numpy()

    backward_psnr, backward_ssim = eval_metrics(output_tensor, backward_tensor)
    backward_lpips = lpips_alex.forward(output_tensor, backward_tensor, normalize=True)
    backward_psnr, backward_ssim = backward_psnr.reshape(batch_size, 7).cpu().numpy(), backward_ssim.reshape(batch_size, 7).cpu().numpy()
    backward_lpips = backward_lpips.reshape(batch_size, 7).detach().cpu().numpy()

    psnrs = parallel * forward_psnr + (1 - parallel) * backward_psnr
    ssims = parallel * forward_ssim + (1 - parallel) * backward_ssim
    lpipss = parallel * forward_lpips + (1 - parallel) * backward_lpips

        # gt1, gt2 = utils.quantize(gt1), utils.quantize(gt2)

        # if parallel:
        #     outimg1, outimg2 = utils.quantize(out1), utils.quantize(out2)
        # else:
        #     outimg2, outimg1 = utils.quantize(out1), utils.quantize(out2)

        # firstname = os.path.splitext(os.path.basename(imgpaths[0][i]))[0]
        # lastname = os.path.splitext(os.path.basename(imgpaths[-1][i]))[0]

        # gt1_fname = foldername + '_' + firstname + '_gt.png'
        # out1_fname = foldername + '_' + firstname + '_out.png'
        # gt2_fname = foldername + '_' + lastname + '_gt.png'
        # out2_fname = foldername + '_' + lastname + '_out.png'

    for b in range(batch_size):
        rows = [imgpaths[ctr_idx][b]]
        rows.extend(list(psnrs[b]))
        df_psnr = df_psnr.append(pd.Series(rows, index=df_psnr.columns), ignore_index=True)

        rows = [imgpaths[ctr_idx][b]]
        rows.extend(list(ssims[b]))
        df_ssim = df_ssim.append(pd.Series(rows, index=df_ssim.columns), ignore_index=True)

        rows = [imgpaths[ctr_idx][b]]
        rows.extend(list(psnrs[b]))
        df_lpips = df_lpips.append(pd.Series(rows, index=df_ssim.columns), ignore_index=True)

    if args.save_img:
        for b in range(batch_size):
            foldername = os.path.basename(os.path.dirname(imgpaths[ctr_idx][b]))
            if not os.path.exists(os.path.join(output_path, foldername)):
                os.mkdir(os.path.join(output_path, foldername))

            if parallel[b][0] == 1:        
                filename1 = os.path.splitext(os.path.basename(imgpaths[0][b]))[0] + '.png'
                filename2 = os.path.splitext(os.path.basename(imgpaths[1][b]))[0] + '.png'
                filename3 = os.path.splitext(os.path.basename(imgpaths[2][b]))[0] + '.png'
                filename4 = os.path.splitext(os.path.basename(imgpaths[3][b]))[0] + '.png'
                filename5 = os.path.splitext(os.path.basename(imgpaths[4][b]))[0] + '.png'
                filename6 = os.path.splitext(os.path.basename(imgpaths[5][b]))[0] + '.png'
                filename7 = os.path.splitext(os.path.basename(imgpaths[6][b]))[0] + '.png'
            
            else:
                filename1 = os.path.splitext(os.path.basename(imgpaths[6][b]))[0] + '.png'
                filename2 = os.path.splitext(os.path.basename(imgpaths[5][b]))[0] + '.png'
                filename3 = os.path.splitext(os.path.basename(imgpaths[4][b]))[0] + '.png'
                filename4 = os.path.splitext(os.path.basename(imgpaths[3][b]))[0] + '.png'
                filename5 = os.path.splitext(os.path.basename(imgpaths[2][b]))[0] + '.png'
                filename6 = os.path.splitext(os.path.basename(imgpaths[1][b]))[0] + '.png'
                filename7 = os.path.splitext(os.path.basename(imgpaths[0][b]))[0] + '.png'
            
            # Comment two lines below if you want to save images
            torchvision.utils.save_image(output_1[b].clone(), os.path.join(save_folder, foldername, filename1))
            torchvision.utils.save_image(output_2[b].clone(), os.path.join(save_folder, foldername, filename2))
            torchvision.utils.save_image(output_3[b].clone(), os.path.join(save_folder, foldername, filename3))
            torchvision.utils.save_image(output_center[b].clone(), os.path.join(save_folder, foldername, filename4))
            torchvision.utils.save_image(output_5[b].clone(), os.path.join(save_folder, foldername, filename5))
            torchvision.utils.save_image(output_6[b].clone(), os.path.join(save_folder, foldername, filename6))
            torchvision.utils.save_image(output_7[b].clone(), os.path.join(save_folder, foldername, filename7))


    if not args.save_img:
        df_psnr.to_csv('{}/results_psnr.csv'.format(output_path))
        df_ssim.to_csv('{}/results_ssim.csv'.format(output_path))
        df_lpips.to_csv('{}/results_lpips.csv'.format(output_path))
