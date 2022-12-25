#!/usr/bin/env python
# -*- coding: utf-8 -*-
""""
Script to extract images from the FaceForensics dataset

Usage:
    # Full cropped dataset
    python extract_images.py
        -i <input path with test/train/val folders>
        -o <output_path>
        --every_nth 1
    # 10 random cropped images of all videos
    python extract_images.py
        -i <input path with test/train/val folders>
        -o <output_path>
        --absolute_num 10
    # Extract from single folder
    python extract_images.py
        -i <input path, i.e. test/val or train folder>
        -o <output_path>
        --absolute_num 10
        -m single_folder
    # Extract from compressed videos but with uncompressed masks
    python extract_images.py
        -i <input path with test/train/val folders>
        -o <output_path>
        --absolute_num 10
        --mask_data_path <input path with test/train/val folders>
    # Full uncropped images + face masks
    python extract_images.py
        -i <input path with test/train/val folders>
        -o <output_path>
        --crop 0
        --every_nth 1
        --return_masks 1
"""
import cv2              # pip install opencv-python
import os
from os.path import join
import argparse
import random
import progressbar      # pip install progressbar2
import numpy as np
from align_and_crop_faces_like_FFHQ import CropLikeFFHQ
import json

def get_non_zero_bb(img):
    # Get non zero elements for mask to get mask area
    a = np.where(img != 255)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox


def create_images_from_single_folder(data_path, output_path,
                                     absolute_num,
                                     every_nth,
                                     crop=1,
                                     output_img_res=None,
                                     return_masks=0,
                                     mask_data_path=None,
                                     **kwargs):
    """
    Extract images from the FaceForensics dataset. Provide either 'absolute_num'
    for an absolute number of images to extract or 'every_nth' if you want to
    extract every nth image of a video.
    If you are only interested in face regions you can crop face regions with
    'crop_faces'. You can specify a 'scale' in order to get a bigger or smaller
    face region.

    :param data_path: contains 'altered', 'original' and maybe 'mask' folder
    :param output_path:
    :param absolute_num: if you want to extract an absolute number of images
    :param every_nth: if you want to extract a percentage of all images
    :param crop: if we crop images to face regions * scale or return full
    images (e.g. for localization)
    :param scale: extension of extracted face region in order to have the full
    face and a little bit of background
    :param mask_data_path: if 'mask' folder is not in data_path
    :param return_masks: if we should also create a folder containing all mask
    images in output_path
    :return:
    """
    # Input folders
    original_data_path = join(data_path, 'original')
    # altered_data_path = join(data_path, 'altered')
    original_filenames = sorted(os.listdir(original_data_path))
    # altered_filenames = sorted(os.listdir(altered_data_path))
    if crop:
        ffhq_style_cropper = CropLikeFFHQ()
    # if crop or return_masks:
    #     if not mask_data_path:
    #         mask_data_path = join(data_path, 'mask')
    #     mask_filenames = sorted(os.listdir(mask_data_path))
    # Check if we have all files
    # assert ([filename[:14] for filename in altered_filenames] ==
    #         [filename[:14] for filename in original_filenames]), \
    #        ("Incorrect number of files in altered and original. " +
    #         "Please check your folders and/or redownload.")
    # if crop or return_masks:
    #     assert ([filename[:14] for filename in altered_filenames] ==
    #             [filename[:14] for filename in original_filenames]), \
    #         ("Incorrect number of files in original/altered and masks." +
    #          "Please check your folders and/or redownload.")

    # Create output folders
    original_images_output_path = join(output_path, 'original')
    os.makedirs(original_images_output_path, exist_ok=True)

    # Progressbar
    bar = progressbar.ProgressBar(max_value=len(original_filenames))
    bar.start()
    exit_next = False
    with open('./dataset/conversion_dict.json', 'r') as f:
        video_file_info = json.load(f)
    # import ipdb; ipdb.set_trace()
    for i in range(len(original_filenames)):
        # Get readers
        original_filename = original_filenames[i]
        youtube_video_id = video_file_info[original_filename[:-4]][:-2]
        original_reader = cv2.VideoCapture(join(original_data_path,
                                                original_filename))
        # Get number of frames
        number_of_frames = int(original_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        if number_of_frames <= 0:
            print('Skipping ' + original_reader + ', invalid number of frames.')
            continue

        # Take video_num_images random frames
        image_frames = list(range(0, number_of_frames))
        assert bool(absolute_num) != bool(every_nth), \
            'You must specify either "absolute num" or "every_nth"'
        if absolute_num is not None and absolute_num > 0:
            absolute_num = min(absolute_num, number_of_frames)
            image_frames = random.sample(image_frames, absolute_num)
        elif absolute_num is None and every_nth > 0:
            image_frames = image_frames[::every_nth]
        image_frames = sorted(image_frames)

        # # Get dimension for scaling
        # width = int(original_reader.get(3))
        # height = int(original_reader.get(4))

        # Frame counter and output filename

        image_counter = 0
        # output_filename_prefix = original_filename.split('.')[0] + '_'

        json_root = os.path.join('faceforensics_original_video_information/Face2Face_video_information',
                                 youtube_video_id, 'faces')
        if not os.path.exists(json_root):
            continue

        output_original_dir = original_filename.split('.')[0]
        os.makedirs(join(original_images_output_path, output_original_dir), exist_ok=True)

        files = os.listdir(json_root)
        # import ipdb; ipdb.set_trace()
        json_files_cpunt = 0
        for file in files:
            json_files_cpunt += 1
        if json_files_cpunt > 1:
            raise ValueError('More than one JSON file. Dont know what to do')

        with open(os.path.join(json_root, file), 'r') as f:
            info_this_vid = json.load(f)
        bounding_boxes = np.array(info_this_vid['faces'])
        original_reader.set(cv2.CAP_PROP_POS_FRAMES, info_this_vid['first frame'] - 1)
        frame_number = info_this_vid['first frame'] - 1

        while original_reader.isOpened() or frame_number >= info_this_vid['last frame']:
            # _, image = altered_reader.read()
            _, original_image = original_reader.read()
            if original_image is None:
                break
            if frame_number == image_frames[0]:
                if crop:
                    # Original image
                    # Primary crop
                    # import ipdb; ipdb.set_trace()
                    # change this to crop
                    # original_image = \
                    #     cv2.rectangle(original_image, tuple(bounding_boxes[image_counter][:2]),
                    #                   tuple(bounding_boxes[image_counter][:2] + bounding_boxes[image_counter][2:]),
                    #                   (255, 0, 0), 3)
                    cntr = bounding_boxes[image_counter][:2] + bounding_boxes[image_counter][2:]//2
                    inflate_hw = bounding_boxes[image_counter][-1] * 3
                    top_left = np.clip((cntr - inflate_hw/2).astype(int), 0, 99999)
                    right_bot = (cntr + inflate_hw / 2).astype(int)
                    # import ipdb; ipdb.set_trace()
                    original_image = original_image[top_left[1]:min(right_bot[1], original_image.shape[0]),
                                     top_left[0]:min(right_bot[0], original_image.shape[1]), :]

                    try:
                        original_image, _ = ffhq_style_cropper.align(original_image, output_size=output_img_res)
                    except ValueError as e:
                        print(f'multiple faces in frame {frame_number} of video {original_filenames[i]}')
                        print(e)
                        exit_next = True



                # Write to files
                # Altered
                # output_altered_filename = output_filename_prefix + \
                #                           str(image_counter) + '.png'
                # cv2.imwrite(join(altered_images_output_path,
                #                  output_altered_filename), image)
                # Original
                output_original_filename = str(frame_number).zfill(4) + '.png'
                cv2.imwrite(join(original_images_output_path, output_original_dir,
                                 output_original_filename), original_image)

                image_counter += 1
                image_frames.pop(0)
                if exit_next:
                    exit(1)

                if len(image_frames) == 0:
                    break
            # Update frame number
            frame_number += 1

        # Release reader sand update progressbar
        # altered_reader.release()
        original_reader.release()
        bar.update(i)
    bar.finish()


def create_images_from_dataset(data_path, output_path, absolute_num, every_nth,
                               crop=1,
                               output_img_res=None,
                               return_masks=0,
                               mask_data_path=None,
                               **kwargs):
    for folder in os.listdir(data_path):
        if folder in ['test', 'val', 'train']:
            if not os.path.exists(join(data_path, folder)):
                print('Skipping {}'.format(join(data_path, folder)))
            else:
                print(join(data_path, folder))
            if mask_data_path:
                folder_mask_data_path = join(mask_data_path, folder, 'mask')
            else:
                folder_mask_data_path = mask_data_path
            create_images_from_single_folder(data_path=join(data_path, folder),
                                             output_path=join(output_path,
                                                              folder),
                                             absolute_num=absolute_num,
                                             every_nth=every_nth,
                                             crop=crop,
                                             output_img_res=output_img_res,
                                             return_masks=return_masks,
                                             mask_data_path=
                                             folder_mask_data_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract images from the FaceForensics dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode', '-m', default='whole_dataset',
                        help='Either "single_folder" for one folder '
                             '(train/test/val) or "whole_dataset" for all '
                             'folders to extract images from altered/original '
                             'and maybe mask videos.')
    parser.add_argument('--data_path', '-i',
                        help='Path to full FaceForensics dataset or single '
                             'folder (train/val/test)')
    parser.add_argument('--output_path', '-o',
                        help='Output folder for extracted images'
                        )
    parser.add_argument('--absolute_num', type=int, default=None,
                        help='Number of randomly chosen images/frames we create'
                             ' per video. Specify either absolute_num or nth '
                             'image')
    parser.add_argument('--every_nth', type=int, default=None,
                        help='Getting every nth image from all videos. Specify '
                             'either absolute_num or nth image')
    parser.add_argument('--output_img_res', type=int, default=256,
                        help='Scale for cropped output images, i.e. if we '
                             'should take a bigger region around the face or '
                             'not.')
    parser.add_argument('--crop', type=int, default=1,
                        help='If we should crop out face regions or not.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility '
                             '(or not if unspecified).')

    config, _ = parser.parse_known_args()

    # Random seed for reproducibility
    if config.seed is not None:
        random.seed(config.seed)

    kwargs = vars(config)

    if config.mode == 'single_folder':
        create_images_from_single_folder(**kwargs)
    elif config.mode == 'whole_dataset':
        create_images_from_dataset(**kwargs)
    else:
        print('Wrong mode, enter either "single_folder" or "whole_dataset".')
