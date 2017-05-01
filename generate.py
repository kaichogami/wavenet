from model import Wavenet


import numpy as np
import librosa
import tensorflow as tf

audio_frequency = 3000
receptive_seconds = 0.65
filter_width = 2
residual_channels = 2
dilation_channels = 2
skip_channels = 2
quantization_channels = 256
audio_trim_secs = 9

data, _ = librosa.load("first.wav", audio_frequency)

if __name__ == '__main__':
    model = Wavenet(audio_frequency, receptive_seconds, filter_width,
                    residual_channels, dilation_channels, skip_channels,
                    quantization_channels)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        try:
            saver.restore(sess, "./training.ckpt")
            print("loadded scuccesffuly")
        except:
            raise ValueError("couldnt load")

        # test code
        # X  = np.float32(np.random.randint(1, 256,(1,1,1000,1)))
        # encoded = tf.one_hot(X, depth=256, dtype=tf.float32)
        # encoded = tf.reshape(encoded, [1, 1, -1, 256])
        # print(sess.run(model.create_network(encoded)))
        music = sess.run(model.generate(3, data))

    librosa.output.write_wav("/output/example.wav", np.reshape(music, [-1]), audio_frequency)
    print("done writing")
