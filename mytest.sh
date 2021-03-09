
python test.py --dataset gopro --output_path quantitative/jin_GoPro --seq_len 7 --cuda --save_img
# python test.py --dataset gopro --output_path quantitative/jin_GoPro --seq_len 11 --cuda
# python test.py --dataset gopro --output_path quantitative/jin_GoPro --seq_len 13 --cuda

# python test.py --dataset reds --output_path quantitative/paper_REDS --seq_len 5 --cuda --save_img
# python test.py --dataset reds --output_path quantitative/paper_REDS --seq_len 7 --cuda 

python interp_test.py --dataset gopro --output_path quantitative/jin_superslomo_GoPro --seq_len 11 --cuda --save_img
python interp_test.py --dataset gopro --output_path quantitative/jin_superslomo_GoPro --seq_len 13 --cuda --save_img

python interp_test.py --dataset reds --output_path quantitative/jin_superslomo_REDS --seq_len 5 --cuda --save_img


# python test.py --dataset gopro --deblur models/mprnet.pth --output_path quantitative/mprnet_GoPro --seq_len 7 --cuda --save_img
# python test.py --dataset gopro --deblur models/mprnet.pth --output_path quantitative/mprnet_GoPro --seq_len 11 --cuda
# python test.py --dataset gopro --deblur models/mprnet.pth --output_path quantitative/mprnet_GoPro --seq_len 13 --cuda

# python test.py --dataset reds --output_path quantitative/paper_REDS --seq_len 5 --cuda --save_img
# python test.py --dataset reds --output_path quantitative/paper_REDS --seq_len 7 --cuda 
