import numpy as np
import lmdb
import os
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO


class FaceForensicsLMDBDataset(Dataset):

    def __init__(self, lmdb_path, mode=None):
        self.lmdb_path = lmdb_path
        # Delay loading LMDB data until after initialization to avoid "can't pickle Environment Object error"
        self.env = None
        self.txn = None
        self.offset = None
        self.mode = mode
        if self.mode == 'Frame5':
            self.offset = 5
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
        lmdb_data = Image.open(BytesIO(lmdb_data)).convert('RGB')
        return lmdb_data

    def __len__(self):
        if self.env is None:
            self._init_db()
        length = int(self.txn.get('length'.encode('utf-8')).decode('utf-8'))
        self.length = length
        return length

    def _get_data(self, index, offset):
        zero_image_key = f'{index:03d}-{offset:04d}-img'
        zero_image_2d_keypt_key = f'{index:03d}-{offset:04d}-2d'
        zero_image_3d_keypt_key = f'{index:03d}-{offset:04d}-3d'

        zero_img_data = self.read_lmdb_img(zero_image_key)
        zero_img_2d_key_pt = self.read_lmdb_np(zero_image_2d_keypt_key)
        zero_img_3d_key_pt = self.read_lmdb_np(zero_image_3d_keypt_key)

        data = dict()
        data['img'] = zero_img_data
        data['2d_keypt'] = zero_img_2d_key_pt
        data['3d_keypt'] = zero_img_3d_key_pt
        return data

    def __getitem__(self, index):
        data = dict()
        # Delay loading LMDB data until after initialization
        if self.env is None:
            self._init_db()

        data['0'] = self._get_data(index=index, offset=0)

        if self.mode is not None and self.offset is not None:
            data[f'{self.offset}'] = self._get_data(index=index, offset=self.offset)
        print(data)
        return data


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    face_forensics_data_loader = \
        DataLoader(dataset=FaceForensicsLMDBDataset(lmdb_path='/home/pravir/Downloads/deca.lmdb',
                                                                            mode='Frame5'),
                   batch_size=1,
                   shuffle=True)
    data = next(iter(face_forensics_data_loader))

