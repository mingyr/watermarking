import os, sys
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import sonnet as snt

from config import FLAGS
from input_ import Input, Watermark
from model import Upsampler, Downsampler, Blender, Extractor
from utils import Summaries, Metrics, reset_metrics
from optimizer import Adam
# from optimizer import SGD
from tensorflow.python.platform import app

class LossRegression(snt.AbstractModule):
    def __init__(self, name = "loss_regression"):
        super(LossRegression, self).__init__(name = name)

    def _build(self, logits, labels):
        with tf.control_dependencies([tf.assert_equal(tf.rank(labels), tf.rank(logits))]):
            loss = tf.reduce_mean(tf.squared_difference(logits, labels), name = 'loss')

        return loss

def train(data_path, image_upsampler, wm_upsampler, blender, downsampler, extrator, summ):
    data_file = os.path.join(data_path, 'train_images.tfr')
    wm_file = os.path.join(data_path, 'watermark.mat')
    assert os.path.isfile(data_file), "Invalid file name"
    assert os.path.isfile(wm_file), "Invalid file name"

    input_ = Input(FLAGS.batch_size, [FLAGS.img_height, FLAGS.img_width, FLAGS.num_chans])
    images = input_(data_file)
    wm = Watermark(wm_file)()

    image_upsampled = image_upsampler(images)
    wm_upsampled = wm_upsampler(wm)
    image_blended = blender(image_upsampled, wm_upsampled)
    image_downsampled = downsampler(image_blended, training = True)
    wm_extracted = extrator(image_downsampled, training = True) 

    image_loss = LossRegression()(image_downsampled, images)
    wm_loss = LossRegression()(wm_extracted, wm)

    opt = Adam(FLAGS.learning_rate, lr_decay = FLAGS.lr_decay, lr_decay_steps = FLAGS.lr_decay_steps,
               lr_decay_factor = FLAGS.lr_decay_factor)

    train_op = opt(image_loss + wm_loss)

    summ.register('train', 'image_loss', image_loss)
    summ.register('train', 'wm_loss', wm_loss)

    train_summ_op = summ('train')

    return image_loss + wm_loss, train_op, train_summ_op

def val(data_path, image_upsampler, wm_upsampler, blender, downsampler, extrator, summ):
    data_file = os.path.join(data_path, 'test_images.tfr')
    wm_file = os.path.join(data_path, 'watermark.mat')
    assert os.path.isfile(data_file), "Invalid file name"
    assert os.path.isfile(wm_file), "Invalid file name"

    input_ = Input(FLAGS.batch_size, [FLAGS.img_height, FLAGS.img_width, FLAGS.num_chans])
    images = input_(data_file)
    wm = Watermark(wm_file)()

    image_upsampled = image_upsampler(images)
    wm_upsampled = wm_upsampler(wm)
    image_blended = blender(image_upsampled, wm_upsampled)
    image_downsampled = downsampler(image_blended)
    wm_extracted = extrator(image_downsampled)

    image_loss = LossRegression()(image_downsampled, images)
    wm_loss = LossRegression()(wm_extracted, wm)

    summ.register('val', 'image_loss', image_loss)
    summ.register('val', 'wm_loss', wm_loss)

    val_summ_op = summ('val')

    return val_summ_op

def main(unused_argv):
    summ = Summaries()

    if FLAGS.data_dir == '' or not os.path.exists(FLAGS.data_dir):
        raise ValueError('invalid data directory {}'.format(FLAGS.data_dir))

    if FLAGS.output_dir == '':
        raise ValueError('invalid output directory {}'.format(FLAGS.output_dir))
    elif not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)   

    event_log_dir = os.path.join(FLAGS.output_dir, '')
    
    checkpoint_path = os.path.join(FLAGS.output_dir, 'model.ckpt')

    print('Constructing models.')

    dim = [FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.num_chans]
    image_upsampler = Upsampler(dim)
    wm_upsampler = Upsampler([1] + dim[1:])
    image_downsampler = Downsampler(dim)
    blender = Blender(dim)
    extrator = Extractor(dim)

    train_loss, train_op, train_summ_op = \
        train(FLAGS.data_dir, image_upsampler, wm_upsampler, blender, image_downsampler, extrator, summ)
    val_summ_op = val(FLAGS.data_dir, image_upsampler, wm_upsampler, blender, image_downsampler, extrator, summ)

    print('Constructing saver.')
    saver = tf.train.Saver()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to as some of the ops do not have GPU implementations.
    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)

    assert (FLAGS.gpus != ''), 'invalid GPU specification'
    config.gpu_options.visible_device_list = FLAGS.gpus

    # Build an initialization operation to run below.
    init = [tf.global_variables_initializer(), tf.local_variables_initializer()]

    with tf.Session(config = config) as sess:
        sess.run(init)

        writer = tf.summary.FileWriter(event_log_dir, graph = sess.graph)

        # Run training.
        for itr in range(FLAGS.num_iterations):
            cost, _, train_summ_str = sess.run([train_loss, train_op, train_summ_op])
            # Print info: iteration #, cost.
            print(str(itr) + ' ' + str(cost))

            writer.add_summary(train_summ_str, itr)

            if itr % FLAGS.validation_interval == 1:
                # Run through validation set.
                val_summ_str = sess.run(val_summ_op)
                writer.add_summary(val_summ_str, itr)

        tf.logging.info('Saving model.')
        saver.save(sess, checkpoint_path)
        tf.logging.info('Training complete')

if __name__ == '__main__':
    app.run()

