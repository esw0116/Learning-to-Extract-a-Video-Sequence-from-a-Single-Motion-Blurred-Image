from __future__ import print_function
import argparse
import glob, os
import lpips
import pandas as pd
import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from data import create_dataset
from model import centerEsti, F17_N9, F26_N9, F35_N8
from metric_utils import eval_metrics
import utils


# Training settings
parser = argparse.ArgumentParser(description='parser for video prediction')
parser.add_argument('--data_root', type=str, required=True, help='input image directory')
parser.add_argument('--output_path', type=str, required=True, help='directory to save outputs')
parser.add_argument('--seq_len', type=int, required=True, help='Length of sequence')
parser.add_argument('--cuda', action='store_true', help='use cuda')

args = parser.parse_args()

device = 'cuda' if args.cuda else 'cpu'

# load model
model1 = centerEsti()
model2 = F35_N8()
model3 = F26_N9()
model4 = F17_N9()

checkpoint = torch.load('models/center_v3.pth')
checkpoint = checkpoint['state_dict_G']
checkpoint_clone = checkpoint.copy() # We can't mutate while iterating
for key, value in checkpoint_clone.items():
    if key.endswith(('running_mean', 'running_var')):
        del checkpoint[key]
model1.load_state_dict(checkpoint)
'''
checkpoint = torch.load('models/center0.pth')
model1.load_state_dict(checkpoint)
'''
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
val_dataset = create_dataset('gopro', args.data_root, img_type='bin', training=False, seq_len=seq_len)
val_datloader = DataLoader(val_dataset, batch_size=4, num_workers=8, shuffle=False)
tqdm_loader = tqdm.tqdm(val_datloader, ncols=80)

df = pd.DataFrame(columns=['Pos', 'Name', 'PSNR', 'SSIM', 'lpips'])
save_folder = os.path.join(args.output_path, 'GoPro_Imgs')
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

for imgbatch, imgpaths in tqdm_loader:
    imgs = imgbatch.chunk(chunks=seq_len, dim=1)

    img_list = []
    for img in imgs:
        img_list.append(img.squeeze(dim=1).to(device))
    blurred_img = torch.stack(img_list, dim=0).mean(dim=0)

    with torch.no_grad():
        output4 = model1(blurred_img)
        output3_5 = model2(blurred_img, output4)
        output2_6 = model3(blurred_img, output3_5[0], output4, output3_5[1])
        output1_7 = model4(blurred_img, output2_6[0], output3_5[0], output3_5[1], output2_6[1])
    
    output_center = output4
    output_first = output1_7[0]
    output_last = output1_7[-1]

    # Calc Center PSNR
    psnr_ctr, ssim_ctr = eval_metrics(output_center, img_list[ctr_idx])
    lpips_metric_ctr = lpips_alex.forward(output_center, img_list[ctr_idx])
    for i in range(psnr_ctr.shape[0]):
        foldername = os.path.basename(os.path.dirname(imgpaths[ctr_idx][i]))
        row = ['Center', foldername + '_' + os.path.basename(imgpaths[ctr_idx][i])[:-4], psnr_ctr[i].item(), ssim_ctr[i].item(), lpips_metric_ctr[i].item()]
        df = df.append(pd.Series(row, index=df.columns), ignore_index=True)

        foldername = os.path.basename(os.path.dirname(imgpaths[ctr_idx][i]))
        centername = os.path.splitext(os.path.basename(imgpaths[ctr_idx][i]))[0]
        out0, gt0 = utils.quantize(output_center[i]), utils.quantize(img_list[ctr_idx][i])

        gt0_fname = foldername + '_' + centername + '_gt.png'
        out0_fname = foldername + '_' + centername + '_out.png'

        torchvision.utils.save_image(out0, os.path.join(save_folder, out0_fname), normalize=True, range=(0,255))
        torchvision.utils.save_image(gt0, os.path.join(save_folder, gt0_fname), normalize=True, range=(0,255))

    # Calc First, Last PSNR
    gt_batch1, gt_batch2, out_batch1, out_batch2  = img_list[0], img_list[-1], output_first, output_last
        
    for i, (gt1, gt2, out1, out2) in enumerate(zip(gt_batch1, gt_batch2, out_batch1, out_batch2)):
        foldername = os.path.basename(os.path.dirname(imgpaths[ctr_idx][i]))

        gt1, gt2, out1, out2 = gt1.unsqueeze(0), gt2.unsqueeze(0), out1.unsqueeze(0), out2.unsqueeze(0)
        psnr11, ssim11 = eval_metrics(out1, gt1)
        psnr12, ssim12 = eval_metrics(out1, gt2)
        psnr21, ssim21 = eval_metrics(out2, gt1)
        psnr22, ssim22 = eval_metrics(out2, gt2)

        psnr_parallel = psnr11 + psnr22
        psnr_crossed  = psnr12 + psnr21

        if psnr_parallel > psnr_crossed:
            psnr = psnr_parallel / 2 if psnr_ctr[i] == 0 else (psnr_parallel + psnr_ctr[i]) / 3
            ssim = (ssim11 + ssim22) / 2 if psnr_ctr[i] == 0 else (ssim11 + ssim22 + ssim_ctr[i]) / 3
            parallel = True
            lpips_metric11 = lpips_alex.forward(out1, gt1)
            lpips_metric22 = lpips_alex.forward(out2, gt2)

            row = ['First', foldername + '_' + os.path.basename(imgpaths[0][i])[:-4], psnr11.item(), ssim11.item(), lpips_metric11.item()]
            df = df.append(pd.Series(row, index=df.columns), ignore_index=True)
            row = ['Last', foldername + '_' + os.path.basename(imgpaths[-1][i])[:-4], psnr22.item(), ssim22.item(), lpips_metric22.item()]
            df = df.append(pd.Series(row, index=df.columns), ignore_index=True)

        else:
            psnr = psnr_crossed / 2 if psnr_ctr[i] == 0 else (psnr_crossed + psnr_ctr[i]) / 3
            ssim = (ssim12 + ssim21) / 2 if psnr_ctr[i] == 0 else (ssim12 + ssim21 + ssim_ctr[i]) / 3
            parallel = False
            lpips_metric12 = lpips_alex.forward(out1, gt2)
            lpips_metric21 = lpips_alex.forward(out2, gt1)

            # Filename is fit to GT
            row = ['Last', foldername + '_' + os.path.basename(imgpaths[-1][i])[:-4], psnr12.item(), ssim12.item(), lpips_metric12.item()]
            df = df.append(pd.Series(row, index=df.columns), ignore_index=True)
            row = ['First', foldername + '_' + os.path.basename(imgpaths[0][i])[:-4], psnr21.item(), ssim21.item(), lpips_metric21.item()]
            df = df.append(pd.Series(row, index=df.columns), ignore_index=True)

        gt1, gt2 = utils.quantize(gt1), utils.quantize(gt2)

        if parallel:
            outimg1, outimg2 = utils.quantize(out1), utils.quantize(out2)
        else:
            outimg2, outimg1 = utils.quantize(out1), utils.quantize(out2)

        firstname = os.path.splitext(os.path.basename(imgpaths[0][i]))[0]
        lastname = os.path.splitext(os.path.basename(imgpaths[-1][i]))[0]

        gt1_fname = foldername + '_' + firstname + '_gt.png'
        out1_fname = foldername + '_' + firstname + '_out.png'
        gt2_fname = foldername + '_' + lastname + '_gt.png'
        out2_fname = foldername + '_' + lastname + '_out.png'

        torchvision.utils.save_image(outimg1, os.path.join(save_folder, out1_fname), normalize=True, range=(0,255))
        torchvision.utils.save_image(gt1, os.path.join(save_folder, gt1_fname), normalize=True, range=(0,255))

        torchvision.utils.save_image(outimg2, os.path.join(save_folder, out2_fname), normalize=True, range=(0,255))
        torchvision.utils.save_image(gt2, os.path.join(save_folder, gt2_fname), normalize=True, range=(0,255))
        
    '''
        if args.cuda:
        output1 = output1_7[0].cpu()
        output2 = output2_6[0].cpu()
        output3 = output3_5[0].cpu()
        output4 = output4.cpu()
        output5 = output3_5[1].cpu()
        output6 = output2_6[1].cpu()
        output7 = output1_7[1].cpu()
    else:
        output1 = output1_7[0]
        output2 = output2_6[0]
        output3 = output3_5[0]
        output4 = output4
        output5 = output3_5[1]
        output6 = output2_6[1]
        output7 = output1_7[1]

    inputFilename = os.path.basename(inputFile)

    output_data = output1.data[0]*255
    utils.save_image(os.path.join(args.output, inputFilename[:-4] + '-esti1' + inputFilename[-4:]), output_data)                
    output_data = output2.data[0]*255
    utils.save_image(os.path.join(args.output, inputFilename[:-4] + '-esti2' + inputFilename[-4:]), output_data)                
    output_data = output3.data[0]*255
    utils.save_image(os.path.join(args.output, inputFilename[:-4] + '-esti3' + inputFilename[-4:]), output_data)
    output_data = output4.data[0]*255
    utils.save_image(os.path.join(args.output, inputFilename[:-4] + '-esti4' + inputFilename[-4:]), output_data)
    output_data = output5.data[0]*255
    utils.save_image(os.path.join(args.output, inputFilename[:-4] + '-esti5' + inputFilename[-4:]), output_data)
    output_data = output6.data[0]*255
    utils.save_image(os.path.join(args.output, inputFilename[:-4] + '-esti6' + inputFilename[-4:]), output_data)
    output_data = output7.data[0]*255
    utils.save_image(os.path.join(args.output, inputFilename[:-4] + '-esti7' + inputFilename[-4:]), output_data)
    '''
df.to_csv('{}/results.csv'.format(args.output_path))
