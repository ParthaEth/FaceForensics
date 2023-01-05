# This code is a copy from 'https://github.com/rosinality/style-based-gan-pytorch/blob/master/prepare_data.py'
# hence the licence as given in 'https://github.com/rosinality/style-based-gan-pytorch/blob/master/LICENSE' applies
import os
import argparse
from io import BytesIO
import multiprocessing

from PIL import Image
import lmdb
from tqdm import tqdm
import numpy as np


def prepare(transaction, flame_dataset_folder, original_images_path):

    total = 0
    video_indices = os.listdir(flame_dataset_folder)

    for video_idx in tqdm(video_indices):
        frame_indices = os.listdir(os.path.join(flame_dataset_folder, video_idx))

        num_frames = str(len(frame_indices))

        if num_frames == '0':
            continue
        num_frames_key = f'{video_idx}-num_frames'.encode('utf-8')
        transaction.put(num_frames_key, num_frames.encode('utf-8'))

        for frame_idx in frame_indices:

            org_image_key = f'{video_idx}-{frame_idx}-org_img'.encode('utf-8')
            org_img_path = os.path.join(original_images_path, video_idx, f'{frame_idx}.png')
            org_img = Image.open(org_img_path).convert('RGB')
            org_img_buffer = BytesIO()
            org_img.save(org_img_buffer, format='png')
            org_img_str = org_img_buffer.getvalue()

            flame_rendering_key = f'{video_idx}-{frame_idx}-flame_img'.encode('utf-8')
            flame_rendering_img_path = os.path.join(flame_dataset_folder,
                                                    video_idx, frame_idx, f'{frame_idx}_recon.png')
            flame_rendering_img = Image.open(flame_rendering_img_path).convert('RGB')
            flame_img_buffer = BytesIO()
            flame_rendering_img.save(flame_img_buffer, format='png')
            flame_img_str = flame_img_buffer.getvalue()

            # wbg -> with background
            flame_rendering_wbg_key = f'{video_idx}-{frame_idx}-flame_img_wbg'.encode('utf-8')
            flame_rendering_img_wbg_path = os.path.join(flame_dataset_folder,
                                                    video_idx, frame_idx, f'{frame_idx}_recon_wbg.png')
            flame_rendering_img_wbg = Image.open(flame_rendering_img_wbg_path).convert('RGB')
            flame_img_wbg_buffer = BytesIO()
            flame_rendering_img_wbg.save(flame_img_wbg_buffer, format='png')
            flame_img_wbg_str = flame_img_wbg_buffer.getvalue()

            two_dim_key = f'{video_idx}-{frame_idx}-2d'.encode('utf-8')
            two_dim_path = os.path.join(flame_dataset_folder, video_idx, frame_idx, f'{frame_idx}_kpt2d.txt')
            two_dim_array = np.loadtxt(two_dim_path)
            two_dim_buffer = BytesIO()
            np.save(two_dim_buffer, two_dim_array)
            two_dim_str = two_dim_buffer.getvalue()

            three_dim_key = f'{video_idx}-{frame_idx}-3d'.encode('utf-8')
            three_dim_path = os.path.join(flame_dataset_folder, video_idx, frame_idx, f'{frame_idx}_kpt3d.txt')
            three_dim_array = np.loadtxt(three_dim_path)
            three_dim_buffer = BytesIO()
            np.save(three_dim_buffer, three_dim_array)
            three_dim_str = three_dim_buffer.getvalue()

            transaction.put(org_image_key, org_img_str)
            transaction.put(flame_rendering_key, flame_img_str)
            transaction.put(flame_rendering_wbg_key, flame_img_wbg_str)
            transaction.put(two_dim_key, two_dim_str)
            transaction.put(three_dim_key, three_dim_str)

            # print(flame_rendering_key, flame_rendering_img_path)
            # print(two_dim_key, two_dim_path)
            # print(three_dim_key, three_dim_path)

    total += 1
    transaction.put('length'.encode('utf-8'), str(total).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--n_worker', type=int, default=1)
    parser.add_argument('--dataset_path', type=str, default='/home/pravir/Downloads/FACEFORENSICS/frames/train')
    parser.add_argument('--output_path', type=str, default='/home/pravir/Downloads/deca.lmdb')

    args = parser.parse_args()

    flame_dataset_path = os.path.join(args.dataset_path, 'deca')
    original_image_path = os.path.join(args.dataset_path, 'original')
    output_path = args.output_path

    with lmdb.open(output_path, map_size=1e12, readahead=False) as env:
        with env.begin(write=True) as txn:
            prepare(transaction=txn, flame_dataset_folder=flame_dataset_path, original_images_path=original_image_path)
