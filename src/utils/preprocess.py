import numpy as np
import tensorflow as tf

def get_keypoints(df, img_size=[96, 96]):
    results_dict = {}

    for idx, row in df.iterrows():
        template = {
            'image':{},
            'left_eye':{},
            'right_eye':{},
            'left_eyebrow':{},
            'right_eyebrow':{},
            'mouth':{},
            'nose':{}
        }

        template['image'] = np.reshape(
            [int(num) for num in row['Image'].split(' ')], img_size).tolist()
        template['left_eye']['center'] = [
            row['left_eye_center_x'], row['left_eye_center_y']]
        template['left_eye']['inner_corner'] = [
            row['left_eye_inner_corner_x'], row['left_eye_inner_corner_y']]
        template['left_eye']['outer_corner'] = [
            row['left_eye_outer_corner_x'], row['left_eye_outer_corner_y']]

        template['right_eye']['center'] = [
            row['right_eye_center_x'], row['right_eye_center_y']]
        template['right_eye']['inner_corner'] = [
            row['right_eye_inner_corner_x'], row['right_eye_inner_corner_y']]
        template['right_eye']['outer_corner'] = [
            row['right_eye_outer_corner_x'], row['right_eye_outer_corner_y']]

        template['left_eyebrow']['inner_end'] = [
            row['left_eyebrow_inner_end_x'], row['left_eyebrow_inner_end_y']]
        template['left_eyebrow']['outer_end'] = [
            row['left_eyebrow_outer_end_x'], row['left_eyebrow_outer_end_y']]

        template['right_eyebrow']['inner_end'] = [
            row['right_eyebrow_inner_end_x'], row['right_eyebrow_inner_end_y']]
        template['right_eyebrow']['outer_end'] = [
            row['right_eyebrow_outer_end_x'], row['right_eyebrow_outer_end_y']]

        template['nose']['tip'] = [row['nose_tip_x'], row['nose_tip_y']]

        template['mouth']['left_corner'] = [
            row['mouth_left_corner_x'], row['mouth_left_corner_y']]
        template['mouth']['right_corner'] = [
            row['mouth_right_corner_x'], row['mouth_right_corner_y']]
        template['mouth']['center_top_lip'] = [
            row['mouth_center_top_lip_x'], row['mouth_center_top_lip_y']]
        template['mouth']['center_bottom_lip'] = [
            row['mouth_center_bottom_lip_x'], row['mouth_center_bottom_lip_y']]
        
        results_dict[idx] = template

    return results_dict

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image, key_pts):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
      'image': _bytes_feature(image),
      'key_pts': _bytes_feature(key_pts)
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
