import os, sys
import numpy as np
import tensorflow as tf
import sonnet as snt

from config import FLAGS
from input_ import Image, Watermark, Enhance
from model import Upsampler, Downsampler, Blender, Extractor
from utils import draw_image
from tensorflow.python.platform import app

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Mask(snt.AbstractModule):
    def __init__(self, height, width, rank, name = 'mask'):
        super(Mask, self).__init__(name = name)
        self._mask = np.zeros([height, width], np.float32)
        self._rank = rank
        
    def _build(self):
        for i in range(self._rank):
            self._mask[i, 0:(self._rank - i)] = 1
            self._mask[i, -(self._rank - i):] = 1
            self._mask[-(self._rank - i), 0:(i+1)] = 1
            self._mask[-(self._rank - i), -(i+1):] = 1
    
            outputs = tf.convert_to_tensor(self._mask)
            
        return outputs


class FreqImage(snt.AbstractModule):
    def __init__(self, mask, name='freq_clip_image'):
        super(FreqClipImage, self).__init__(name=name)
        self._mask = mask

    def _build(self, inputs):
        outputs = tf.cast(inputs, tf.complex64)
        output_list = [tf.fft2d(e) for e in tf.unstack(outputs, axis=(-1))]
        
        output_list = [e * self._mask for e in output_list]
        image_list = [tf.abs(tf.ifft2d(e)) for e in output_list]
        
        image_list = [tf.clip_by_value(e, 0.0, 255.0) for e in image_list]
        outputs = tf.stack(image_list, axis=-1)
        
        with tf.control_dependencies([tf.equal(tf.shape(inputs), tf.shape(outputs))]):
            outputs = tf.identity(outputs)    

        return outputs

def test_mask():
    mask = Mask(12, 12, 5)()

    with tf.Session() as (sess):
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        v = sess.run(mask)
        np.set_printoptions(threshold = sys.maxsize)
        
        print(v)

def test_freqimage():
    from config import FLAGS
    
    original_image = Image('/data/yuming/watermark-data/image_paths.mat', 10)()
    mask = Mask(FLAGS.img_height, FLAGS.img_width, 64)()
    
    mask = tf.cast(mask, tf.complex64)
    
    freqimage = FreqImage(mask)
    blurred_image = freqimage(original_image)

    writer = tf.summary.FileWriter('model-output', tf.get_default_graph())
    with tf.Session() as (sess):
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        original_image_val, blurred_image_val = sess.run([original_image, blurred_image])
        
        images = [{'data':np.squeeze(original_image_val[0, :, :, :].astype(np.uint8)), 'title': 'original image'},
                  {'data':np.squeeze(blurred_image_val[0, :, :, :].astype(np.uint8)), 'title': 'blurred image'}]
        image_str = draw_image(images)
        
        writer.add_summary(image_str, global_step=0)
    writer.close()


def main(unused_argv):
    if FLAGS.checkpoint_dir == '' or not os.path.exists(FLAGS.checkpoint_dir):
        raise ValueError('invalid checkpoint directory {}'.format(FLAGS.checkpoint_dir))

    checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, '')

    if FLAGS.output_dir == '':
        raise ValueError('invalid output directory {}'.format(FLAGS.output_dir))
    elif not os.path.exists(FLAGS.output_dir):
        assert FLAGS.output_dir != FLAGS.checkpoint_dir
        os.makedirs(FLAGS.output_dir)

    print('reconstructing models and inputs.')
    image = Image('/data/yuming/watermark-data/image_paths.mat', FLAGS.image_seq)()
    wm = Watermark('/data/yuming/watermark-data/watermark.mat')()

    dim = [1, FLAGS.img_height, FLAGS.img_width, FLAGS.num_chans]
    image_upsampler = Upsampler(dim)
    wm_upsampler = Upsampler([1] + dim[1:])
    downsampler = Downsampler(dim)
    blender = Blender(dim)
    extrator = Extractor(dim)

    image_upsampled = image_upsampler(image)
    wm_upsampled = wm_upsampler(wm)
    image_blended = blender(image_upsampled, wm_upsampled)
    image_downsampled = downsampler(image_blended)
    
    mask = Mask(FLAGS.img_height, FLAGS.img_width, 80)()    
    mask = tf.cast(mask, tf.complex64)
    freqimage = FreqImage(mask)
    
    image_freqfiltered = freqimage(image_downsampled)
    wm_extracted = extrator(image_freqfiltered)
    
    enhance = Enhance(sharpen = True)
    wm_extracted = enhance(wm_extracted)
    
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(FLAGS.output_dir, tf.get_default_graph())    

    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    assert (FLAGS.gpus != ''), 'invalid GPU specification'
    config.gpu_options.visible_device_list = FLAGS.gpus

    with tf.Session(config = config) as sess:
        sess.run(tf.local_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        wm_val, image_downsampled_val, image_freqfiltered_val, wm_extracted_val = \
            sess.run([wm, image_downsampled, image_freqfiltered, wm_extracted])

        images = [{'data': np.squeeze(image_downsampled_val[0, :, :, :].astype(np.uint8)), 'title': "watermarked image"},
                  {'data': np.squeeze(image_freqfiltered_val[0, :, :, :].astype(np.uint8)), 'title': "filtered image"},
                  {'data': np.squeeze(wm_val[0, :, :, :].astype(np.uint8)), 'title': "original watermark"},
                  {'data': np.squeeze(wm_extracted_val[0, :, :, :].astype(np.uint8)), 'title': "extracted watermark"}]

        image_str = draw_image(images)
        writer.add_summary(image_str, global_step = 0)
        
        np.set_printoptions(threshold = sys.maxsize)
        print(np.squeeze(wm_extracted_val))

    writer.close()


if __name__ == '__main__':
    # test_mask()
    # test_freqclipimage()

    tf.app.run()

