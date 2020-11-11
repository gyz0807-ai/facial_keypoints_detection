import numpy as np

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
