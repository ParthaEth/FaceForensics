import random

import numpy as np
import lmdb
import os
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
import torch


class FaceForensicsLMDBDataset(Dataset):

    def __init__(self, lmdb_path, offset_mode=None, transforms=None, flame_type='flame_img'):  # flame_img_wbg
        self.lmdb_path = lmdb_path
        # Delay loading LMDB data until after initialization to avoid "can't pickle Environment Object error"
        self.env = None
        self.txn = None
        self.offset = None
        self.offset_mode = offset_mode
        self.transforms = transforms
        self.flame_type = flame_type
        if self.offset_mode == 'Random':
            self.offset = -1
        # Add if-else-if conditions here

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_path, subdir=os.path.isdir(self.lmdb_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.txn = self.env.begin()

    def read_lmdb_np(self, key):
        lmdb_data = self.txn.get(key.encode('utf-8'))
        lmdb_data = np.frombuffer(lmdb_data)
        return lmdb_data

    def read_lmdb_img(self, key):
        lmdb_data = self.txn.get(key.encode('utf-8'))
        lmdb_data = Image.open(BytesIO(lmdb_data)) # .convert('RGB')  # TODO check this convert func
        if self.transforms is not None:
            lmdb_data = self.transforms(lmdb_data)
        return lmdb_data

    def __len__(self):
        if self.env is None:
            self._init_db()
        length = int(self.txn.get('length'.encode('utf-8')).decode('utf-8'))
        self.length = length
        return length

    def _get_data(self, index, offset):
        org_image_key = f'{index:03d}-{offset:04d}-org_img'
        flame_image_key = f'{index:03d}-{offset:04d}-{self.flame_type}'

        # image_2d_keypt_key = f'{index:03d}-{offset:04d}-2d'
        # image_3d_keypt_key = f'{index:03d}-{offset:04d}-3d'

        org_img_data = self.read_lmdb_img(org_image_key)
        flame_img_data = self.read_lmdb_img(flame_image_key)

        # img_2d_key_pt = self.read_lmdb_np(image_2d_keypt_key)
        # img_3d_key_pt = self.read_lmdb_np(image_3d_keypt_key)

        data = dict()
        data['org_img'] = org_img_data
        data[self.flame_type] = flame_img_data

        # data['2d_keypt'] = img_2d_key_pt
        # data['3d_keypt'] = img_3d_key_pt
        return data

    def _package_data_for_gif(self, data, index, offset):
        offset_key = f'{offset}'
        nth_frame_org_img = data[offset_key]['org_img']
        conditions = torch.cat((data['0']['org_img'], data[offset_key][self.flame_type]), dim=0)
        return nth_frame_org_img, conditions, 0, index

    def _get_num_frames(self, index):
        num_frames_key = f'{index:03d}-num_frames'.encode('utf-8')
        num_frames = int(self.txn.get(num_frames_key).decode('utf-8'))
        return num_frames

    def __getitem__(self, index):
        data = dict()
        # Delay loading LMDB data until after initialization
        if self.env is None:
            self._init_db()

        data['0'] = self._get_data(index=index, offset=0)

        if self.offset_mode is not None and self.offset == -1:
            random_offset = random.randint(1, (self._get_num_frames(index=index) - 1)) * 5
            data[f'{random_offset}'] = self._get_data(index=index, offset=random_offset)
        # print(data)
        gif_data = self._package_data_for_gif(data=data, index=index, offset=random_offset)

        return gif_data


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms

    img_transforms = transforms.Compose([transforms.ToTensor()])
    face_forensics_data_loader = \
        DataLoader(dataset=FaceForensicsLMDBDataset(lmdb_path='/home/pravir/Downloads/deca.lmdb',
                                                    offset_mode='Random', transforms=img_transforms),
                   batch_size=1,
                   shuffle=True)
    data = next(iter(face_forensics_data_loader))

