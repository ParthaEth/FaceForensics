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


def prepare(transaction, dataset_folder):

    total = 0
    folders = os.listdir(dataset_folder)

    for folder in tqdm(folders):
        sub_folders = os.listdir(os.path.join(dataset_folder, folder))

        for sub_folder in sub_folders:

            img_key = f'{folder}-{sub_folder}-img'.encode('utf-8')
            img_path = os.path.join(dataset_folder, folder, sub_folder, f'{sub_folder}_recon.png')

            img = Image.open(img_path).convert('RGB')
            img_buffer = BytesIO()
            img.save(img_buffer, format='png')
            img_str = img_buffer.getvalue()

            two_dim_key = f'{folder}-{sub_folder}-2d'.encode('utf-8')
            two_dim_path = os.path.join(dataset_folder, folder, sub_folder, f'{sub_folder}_kpt2d.txt')
            two_dim_array = np.loadtxt(two_dim_path)
            two_dim_buffer = BytesIO()
            np.save(two_dim_buffer, two_dim_array)
            two_dim_str = two_dim_buffer.getvalue()

            three_dim_key = f'{folder}-{sub_folder}-3d'.encode('utf-8')
            three_dim_path = os.path.join(dataset_folder, folder, sub_folder, f'{sub_folder}_kpt3d.txt')
            three_dim_array = np.loadtxt(three_dim_path)
            three_dim_buffer = BytesIO()
            np.save(three_dim_buffer, three_dim_array)
            three_dim_str = three_dim_buffer.getvalue()

            transaction.put(img_key, img_str)
            transaction.put(two_dim_key, two_dim_str)
            transaction.put(three_dim_key, three_dim_str)

            # print(img_key, img_path)
            # print(two_dim_key, two_dim_path)
            # print(three_dim_key, three_dim_path)

    total += 1
    transaction.put('length'.encode('utf-8'), str(total).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--n_worker', type=int, default=1)
    parser.add_argument('--input_path', type=str, default='/home/pravir/Downloads/deca')
    parser.add_argument('--output_path', type=str, default='/home/pravir/Downloads/deca.lmdb')

    args = parser.parse_args()

    dataset_path = args.input_path
    output_path = args.output_path

    with lmdb.open(output_path, map_size=1e12, readahead=False) as env:
        with env.begin(write=True) as txn:
            prepare(txn, dataset_path)
