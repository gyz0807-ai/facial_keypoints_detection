import json
import pandas as pd
from absl import app, logging, flags
from src.utils.preprocess import get_keypoints

FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', './data/training.csv', 'Path to training data')
flags.DEFINE_list('img_size', ['96', '96'], 'Size of training image')

def main(argv=None):
    img_size = [int(n) for n in FLAGS.img_size]

    logging.info('Loading data...')
    df_train = pd.read_csv(FLAGS.data_path)

    logging.info('Parsing data...')
    data_dict = get_keypoints(df_train, img_size=img_size)

    logging.info('Saving data...')

    return

if __name__ == '__main__':
    app.run(main)
