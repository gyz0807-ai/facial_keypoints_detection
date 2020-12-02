import os
import json
import numpy as np
import tensorflow as tf
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from absl import app, logging, flags
from imgaug.augmentables import Keypoint, KeypointsOnImage
from utils.preprocess import _bytes_feature, serialize_example

FLAGS = flags.FLAGS
flags.DEFINE_string('processed_data_dir', './data/processed/processed_data.json', 'Path to processed data')
flags.DEFINE_list('brightness_range', ['0.5', '1.0'], 'Range for brightness change')
flags.DEFINE_list('rotation_range', ['-90', '90'], 'Range for rotation change')
flags.DEFINE_list('scale_range', ['0.5', '1'], 'Range for scale change')
flags.DEFINE_integer('num_augmentations', 10, 'Number of augmentations for each pic')
flags.DEFINE_string('save_to_path', './data/processed/augmentations', 'Path to save augmentation dataset')

def main(argv=None):
    with open(FLAGS.processed_data_dir, 'r') as f:
        data_dict = json.load(f)
    Path(FLAGS.save_to_path).mkdir(parents=True, exist_ok=True)

    brightness_range = tuple([float(n) for n in FLAGS.brightness_range])
    rotation_range = tuple([float(n) for n in FLAGS.rotation_range])
    scale_range = tuple([float(n) for n in FLAGS.scale_range])
    seq = iaa.Sequential([
        iaa.Multiply(brightness_range),
        iaa.Affine(
            rotate=rotation_range,
            scale=scale_range)])

    for n in tqdm(range(FLAGS.num_augmentations+1), desc='Epochs', total=FLAGS.num_augmentations+1):
        fnm = os.path.join(FLAGS.save_to_path, 'augmentation_{}.tfrecord'.format(n))
        with tf.io.TFRecordWriter(fnm) as writer:
            for info in tqdm(data_dict.values(), desc='Pics', total=len(data_dict), leave=False):
                image = np.array(info['image']).astype(np.float32)
                key_pts = KeypointsOnImage([
                    Keypoint(info['left_eye']['center'][0], info['left_eye']['center'][1]),
                    Keypoint(info['left_eye']['inner_corner'][0], info['left_eye']['inner_corner'][1]),
                    Keypoint(info['left_eye']['outer_corner'][0], info['left_eye']['outer_corner'][1]),
                    Keypoint(info['right_eye']['center'][0], info['right_eye']['center'][1]),
                    Keypoint(info['right_eye']['inner_corner'][0], info['right_eye']['inner_corner'][1]),
                    Keypoint(info['right_eye']['outer_corner'][0], info['right_eye']['outer_corner'][1]),
                    Keypoint(info['left_eyebrow']['inner_end'][0], info['left_eyebrow']['inner_end'][1]),
                    Keypoint(info['left_eyebrow']['outer_end'][0], info['left_eyebrow']['outer_end'][1]),
                    Keypoint(info['right_eyebrow']['inner_end'][0], info['right_eyebrow']['inner_end'][1]),
                    Keypoint(info['right_eyebrow']['outer_end'][0], info['right_eyebrow']['outer_end'][1]),
                    Keypoint(info['mouth']['left_corner'][0], info['mouth']['left_corner'][1]),
                    Keypoint(info['mouth']['right_corner'][0], info['mouth']['right_corner'][1]),
                    Keypoint(info['mouth']['center_top_lip'][0], info['mouth']['center_top_lip'][1]),
                    Keypoint(info['mouth']['center_bottom_lip'][0], info['mouth']['center_bottom_lip'][1]),
                    Keypoint(info['nose']['tip'][0], info['nose']['tip'][1])
                ], shape=image.shape)

                if n == 0:
                    image_aug = image
                    key_pts_aug_arr = key_pts.to_xy_array()
                else:
                    image_aug, key_pts_aug = seq(image=image, keypoints=key_pts)
                    key_pts_aug_arr = key_pts_aug.to_xy_array()

                image_aug /= 255
                key_pts_aug_arr[:, 0] /= image_aug.shape[1]
                key_pts_aug_arr[:, 1] /= image_aug.shape[0]
                example = serialize_example(image_aug.tostring(), key_pts_aug_arr.tostring())
                writer.write(example)

    return

if __name__ == '__main__':
    app.run(main)
