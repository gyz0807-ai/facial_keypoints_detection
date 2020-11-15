import os
import json
import pandas as pd
from absl import app, logging, flags
from utils.preprocess import get_keypoints

FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', './data/training.csv', 'Path to training data')
flags.DEFINE_list('img_size', ['96', '96'], 'Size of training image')
flags.DEFINE_string('save_to_path', './data/processed', 'Path to save processed data')

def main(argv=None):
    img_size = [int(n) for n in FLAGS.img_size]

    logging.info('Loading data...')
    df_train = pd.read_csv(FLAGS.data_path)
    data_df_nonull = df_train.loc[df_train.isnull().sum(axis=1) == 0]
    data_df_nonull.reset_index(drop=True, inplace=True)

    logging.info('Parsing data...')
    data_dict = get_keypoints(data_df_nonull, img_size=img_size)

    logging.info('Saving data...')
    with open(os.path.join(FLAGS.save_to_path, 'processed_data.json'), 'w') as f:
        json.dump(data_dict, f)

    return

if __name__ == '__main__':
    app.run(main)
