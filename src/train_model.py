import os
import tensorflow as tf
from tensorflow import keras
from utils.nn import nn_new
from absl import app, logging, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', './data/processed/augmentations',
    'Path to processed tfrecord data')
flags.DEFINE_string('save_as', './models', 'Path to save models')
flags.DEFINE_list('input_size', ['96', '96', '1'], 'Size of input image')
flags.DEFINE_integer('shuffle_buffer_size', 100, 'Shuffle buffer size')
flags.DEFINE_integer('batch_size', 32, 'Batch size for training')
flags.DEFINE_integer('num_epochs', 100, 'Number of epochs')
flags.DEFINE_integer('prefetch_size', 1, 'Number of batches to prefetch')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')

def main(argv=None):
    img_size = [int(n) for n in FLAGS.input_size]
    feature_shapes = {
        'image':img_size,
        'key_pts':[-1]
    }
    feature_desc = {
        'image':tf.io.FixedLenFeature([], tf.string),
        'key_pts':tf.io.FixedLenFeature([], tf.string)
    }
    def _parse_function(example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary above.
        out = tf.io.parse_single_example(example_proto, feature_desc)
        for k, v in out.items():
            out[k] = tf.reshape(tf.io.decode_raw(v, tf.float32), feature_shapes[k])
        return (out['image'], out['key_pts'])

    tf_data_fnms = tf.data.Dataset.list_files(os.path.join(FLAGS.data_path, '*.tfrecord'))
    tf_data = tf.data.TFRecordDataset(tf_data_fnms) \
        .map(_parse_function) \
        .repeat(FLAGS.num_epochs) \
        .shuffle(FLAGS.shuffle_buffer_size) \
        .batch(FLAGS.batch_size) \
        .prefetch(FLAGS.prefetch_size)

    nn_model = nn_new(img_size, is_training=True)
    nn_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=FLAGS.learning_rate),
        loss=keras.losses.mean_squared_error,
        metrics=['mse'])
    nn_model.fit(tf_data)
    nn_model.save(FLAGS.save_as, 'model')
    return

if __name__ == '__main__':
    app.run(main)
