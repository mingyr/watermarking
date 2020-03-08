import os, sys
import numpy as np
import tensorflow as tf
import sonnet as snt

from config import FLAGS
from input_ import Image, Watermark
from model import Upsampler, Downsampler, Blender, Extractor
from utils import draw_image
from tensorflow.python.platform import app
from scipy import io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class FiltImage(snt.AbstractModule):
    def __init__(self, freq, name='filt_image'):
        super(FiltImage, self).__init__(name=name)
        if freq == 'low':
            
            gaussian = np.array([
                [0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625]], dtype = np.float32)
            
            self._filt = np.zeros(gaussian.shape + (3, 3), dtype = np.float32)
            for i in range(gaussian.shape[0]):
                for j in range(gaussian.shape[1]):
                    for k in range(3):
                        for l in range(3):
                            self._filt[i, j, k, l] = gaussian[i, j] if k == l else 0
                            
        elif freq == 'high':
            laplacian = np.array([
                [-0.125, -0.25, -0.125],
                [-0.25, 2.5, -0.25],
                [-0.125, -0.25, -0.125]], dtype = np.float32)

            self._filt = np.zeros(laplacian.shape + (3, 3), dtype = np.float32)
            for i in range(laplacian.shape[0]):
                for j in range(laplacian.shape[1]):
                    for k in range(3):
                        for l in range(3):
                            self._filt[i, j, k, l] = laplacian[i, j] if k == l else 0
        else:
            raise ValueError("Either high-pass filter or low-pass filter")
        
    def _build(self, inputs):
        
        outputs = tf.nn.conv2d(inputs, self._filt, [1, 1, 1, 1], 'SAME')
        outputs = tf.clip_by_value(outputs, 0.0, 255.0)
        
        return outputs

def test_filtimage():
    original_image = Image('/data/yuming/watermark-data/image_paths.mat', 10)()
    filtimage = FiltImage(freq = 'high')
    filtered_image = filtimage(original_image)

    writer = tf.summary.FileWriter('model-output', tf.get_default_graph())
    with tf.Session() as (sess):
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        original_image_val, filtered_image_val = sess.run([original_image, filtered_image])
        images = [{'data':np.squeeze(original_image_val[0, :, :, :].astype(np.uint8)), 'title': 'original image'},
                  {'data':np.squeeze(filtered_image_val[0, :, :, :].astype(np.uint8)), 'title': 'filtered image'}]

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
    # filtimage = FiltImage(freq = 'low')
    filtimage = FiltImage(freq = 'high')

    extrator = Extractor(dim)

    image_upsampled = image_upsampler(image)
    wm_upsampled = wm_upsampler(wm)
    image_blended = blender(image_upsampled, wm_upsampled)
    image_downsampled = downsampler(image_blended)
    image_filtered = filtimage(image_downsampled)
    wm_extracted = extrator(image_filtered)
    
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

        image_downsampled_val, wm_val, image_filtered_val, wm_extracted_val = \
            sess.run([image_downsampled, wm, image_filtered, wm_extracted])

        '''
        images = [{'data': np.squeeze(image_downsampled_val[0, :, :, :].astype(np.uint8)), 'title': "watermarked image"},
                  # {'data': np.squeeze(image_filtered_val[0, :, :, :].astype(np.uint8)), 'title': '(high-pass) filtered image'},
                  {'data': np.squeeze(image_filtered_val[0, :, :, :].astype(np.uint8)), 'title': '(low-pass) filtered image'},
                  {'data': np.squeeze(wm_val[0, :, :, :].astype(np.uint8)), 'title': "original watermark"},
                  {'data': np.squeeze(wm_extracted_val[0, :, :, :].astype(np.uint8)), 'title': 'extracted watermark'}]
        '''

        images = [{'data': np.squeeze(image_downsampled_val[0, :, :, :].astype(np.uint8)), 'title': ''},
                  {'data': np.squeeze(image_filtered_val[0, :, :, :].astype(np.uint8)), 'title': ''},
                  {'data': np.squeeze(wm_val[0, :, :, :].astype(np.uint8)), 'title': ''},
                  {'data': np.squeeze(wm_extracted_val[0, :, :, :].astype(np.uint8)), 'title': ''}]
        
        image_str = draw_image(images)
        writer.add_summary(image_str, global_step = 0)

        io.savemat(os.path.join(FLAGS.output_dir, "filt-test-data.mat"), 
                   {"wm": wm_val, 'wm_extracted': wm_extracted_val})
        
    writer.close()

if __name__ == '__main__':
    # test_filtimage()

    tf.app.run()

