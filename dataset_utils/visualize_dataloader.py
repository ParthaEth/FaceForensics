import torchvision.utils

from dataset_utils.face_forensic_data_loader import FaceForensicsLMDBDataset
from PIL import Image
from torchvision.utils import save_image


def _process_gif_batch(batch, global_batch):
    nth_img_index = 0
    cond_index = 1

    for img, cond in zip(batch[nth_img_index], batch[cond_index]):
        global_batch.append(img)
        global_batch.append(cond[0:3])
        global_batch.append(cond[3:])
    # global_batch.append(cond_index)


def visualize_data_loader(data_loader, output_image_path, n_row):
    global_batch = []

    for batch in data_loader:
        _process_gif_batch(batch=batch, global_batch=global_batch)
    torchvision.utils.save_image(global_batch, fp=output_image_path, nrow=n_row*3)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms

    img_transforms = transforms.Compose([transforms.ToTensor()])
    face_forensics_data_loader = \
        DataLoader(dataset=FaceForensicsLMDBDataset(lmdb_path='/home/pravir/Downloads/deca.lmdb',
                                                    offset_mode='Random_GIF', transforms=img_transforms,
                                                    flame_type='flame_img'),
                   batch_size=1,
                   shuffle=False)
    face_forensics_data_loader = iter(face_forensics_data_loader)
    n_row = 1

    visualize_data_loader(data_loader=face_forensics_data_loader, output_image_path='/home/pravir/Downloads/deca.png',
                          n_row=n_row)