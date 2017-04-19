"""This model uses floyd cloud computers to train. Works only on the dataset
available in http://opihi.cs.uvic.ca/sound/genres.tar.gz
To run this module using floyd

* first extract the contents of the data set and store it in /output/music by
  running a extracting tarball script. 
      ```floyd run "download_extract.py" ```
* Note the output ID and use it for training.
      ```floyd run --data objectID --env tensorflow "python train.py" ```
* The output save files could be used for generating or continuing the training.
"""

from model import Wavenet

import os
import tarfile
import tensorflow as tf
import numpy as np
import librosa

LEARNING_RATE = 0.003
TRAINING_ITER = 10
MOMENTUM = 0.9

# runs on 3000 sampling rate and 15 seconds length
audio_frequency = 3000
receptive_seconds = 1
filter_width = 2
residual_channels = 2
dilation_channels = 2
skip_channels = 2
quantization_channels = 256
audio_trim_secs = 9

# works for only http://opihi.cs.uvic.ca/sound/genres.tar.gz
def _get_data_files(genre):
    list_files = os.listdir(("/input/music/genres/{0}".format(genre)))
    for i in xrange(len(list_files)):
        list_files[i] = ("/input/music/genres/{0}/{1}").format(genre, list_files[i])

    return list_files

def _get_music_array(data):
    y, sr = librosa.load(data, audio_frequency)
    return y[0:audio_frequency * audio_trim_secs]

if __name__ == '__main__':
    # change it into any genere you want, available
    file_list = _get_data_files("classical")
    model = Wavenet(audio_frequency, receptive_seconds, filter_width,
                    residual_channels, dilation_channels, skip_channels,
                    quantization_channels)

    # Make sure batch=1 
    X = tf.placeholder(tf.int32, shape=[1, 1, audio_frequency * audio_trim_secs , 1])
    # define loss and optimizer
    loss = model.loss(X)
    optimizer = tf.train.MomentumOptimizer(LEARNING_RATE, MOMENTUM).minimize(loss)

    saver = tf.train.Saver()

    config = tf.ConfigProto(log_device_placement=False)
    with tf.Session(config=config) as sess:
        try:
            saver.restore(sess, "/input/training.ckpt")
        except:
            init = tf.global_variables_initializer()
            sess.run(init)
        for i in xrange(TRAINING_ITER):
            for j in xrange(len(file_list)):
                data = np.reshape(_get_music_array(file_list[j]),
                                  [1, 1, audio_frequency * audio_trim_secs, 1])
                sess.run(optimizer, feed_dict={X : data})
    save_path = saver.save(sess, "/output/training.ckpt")
    print("save in {0}").format(save_path)

    print("done")
