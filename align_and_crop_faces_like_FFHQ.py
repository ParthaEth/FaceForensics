import os

import numpy as np
import argparse
import PIL.Image
import face_alignment



class CropLikeFFHQ:
    def __init__(self):
        self.landmarks_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

    def align(self, img, output_size=256, take_largest=True):
        face_landmarks = self.landmarks_detector.get_landmarks(img)
        lf_id = 0
        if len(face_landmarks) > 1:
            if take_largest:
                max_eye_2_eye = 0
                for face_id, flm in enumerate(face_landmarks):
                    lm_eye_left, lm_eye_right, lm_mouth_outer = self.interpreat_landmarks(flm)
                    eye_left = np.mean(lm_eye_left, axis=0)
                    eye_right = np.mean(lm_eye_right, axis=0)
                    eye_to_eye = np.linalg.norm(eye_right - eye_left)
                    if eye_to_eye >= max_eye_2_eye:
                        lf_id = face_id
            else:
                raise ValueError(f'image has multiple faces')
        largest_face_landmarks = face_landmarks[lf_id]
        return self.image_align_68(img, largest_face_landmarks, output_size)

    def image_align_68(self, img, face_landmarks, output_size=256):
        # Align function from FFHQ dataset pre-processing step
        # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

        lm_eye_left, lm_eye_right, lm_mouth_outer = self.interpreat_landmarks(face_landmarks)

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        img = PIL.Image.fromarray(img)
        orig_imag_size = img.size[0]

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink


        # Transform.
        # import ipdb; ipdb.set_trace()
        img = img.transform((output_size, output_size), PIL.Image.QUAD, (quad + 0.5).flatten(),
                            PIL.Image.BILINEAR)
        img = np.array(img)

        crop_attribs = {'quad': quad, 'shrink': shrink, 'orig_imag_size': orig_imag_size, 'output_size': output_size}
        return img, crop_attribs

    def interpreat_landmarks(self, face_landmarks):
        lm = np.array(face_landmarks)
        lm_chin = lm[0: 17, :2]  # left-right
        lm_eyebrow_left = lm[17: 22, :2]  # left-right
        lm_eyebrow_right = lm[22: 27, :2]  # left-right
        lm_nose = lm[27: 31, :2]  # top-down
        lm_nostrils = lm[31: 36, :2]  # top-down
        lm_eye_left = lm[36: 42, :2]  # left-clockwise
        lm_eye_right = lm[42: 48, :2]  # left-clockwise
        lm_mouth_outer = lm[48: 60, :2]  # left-clockwise
        lm_mouth_inner = lm[60: 68, :2]  # left-clockwise
        return lm_eye_left, lm_eye_right, lm_mouth_outer
