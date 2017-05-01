"""Model for wavenet. Defines ops in tensorflow sense"""

import numpy as np
import tensorflow as tf

MIN_DIL = 2
MAX_DIL = 4096

def _dilated_convolution(X, filters, dilation, name):
    """Helper function to carry out dilated convolution

    Parameters
    ==========
    X : tf.Tensor of shape(batch, width, height, in_channels)
        The input data
    filters : tf.Tensor of shape(height, width, in_channels, out_channels)
        The filter tensor
    dilation : int
        the dilation factor
    """
    return tf.nn.atrous_conv2d(X, filters, dilation, "SAME", name)


def _create_variable(name, shape):
    """Helped function to create variables using xavier initialization

    Parameters
    ==========
    name : string
        Then name of the variable
    shape : tuple, list
        The shape of the variable
    """
    return tf.get_variable(name, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())


class Wavenet:
    """Model for Wavenet.
    
    Parameters
    ==========
    audio_frequency : int, secs
        The frequency of the audio
    receptive_seconds : int, secs
        The size of the receptive field in seconds.
    filter_width : int,
        Size of the filter.
    residual_channels : int
        No of filters to learn for residual block.
    dilation_channels : int
        No of filters to learn for dilation block.
    skip_channels : int
        No of filters to learn for skip block.
    quantization_channels : int
        No of channels to encode the audio with
    """

    def __init__(self, audio_frequency, receptive_seconds,
                 filter_width,residual_channels,
                 dilation_channels, skip_channels, quantization_channels):
        self.audio_frequency = audio_frequency
        self.receptive_seconds = receptive_seconds
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.quantization_channels = quantization_channels
        self.dilations = _get_dilations(audio_frequency, receptive_seconds)
        self.variables = self._get_all_variables()
        self.quantization_channels = quantization_channels

    
    def _get_all_variables(self):
        """Helper function to create a dict of all variables
        """

        variables = dict()
        # first causal convolution
        with tf.variable_scope("initial_causal_conv"):
            variables['initial_filter'] = _create_variable("filter",
                                                           [1,1,
                                                            self.quantization_channels,
                                                            self.residual_channels])

        variables['dilated_stack'] = list()
        # Dilated stack dictionary with list of variables
        with tf.variable_scope('dilated_stack'):
             for i, _ in enumerate(self.dilations):
                 current = dict()
                 with tf.variable_scope("dilated_layer_{}".format(i)):
                     current['filter'] = _create_variable(
                         "filter", [1, self.filter_width,
                                    self.residual_channels,
                                    self.dilation_channels])

                     current['gate'] = _create_variable(
                         "gate", [1, self.filter_width,
                                  self.residual_channels,
                                  self.dilation_channels])

                     current['skip'] = _create_variable(
                         "skip", [1, self.filter_width,
                                  self.dilation_channels,
                                  self.skip_channels])
                     variables['dilated_stack'].append(current)

        with tf.variable_scope('post_processing'):
            variables['post_1'] = _create_variable(
                "post_1", [1, 1, self.skip_channels, self.skip_channels])
            variables['post_2'] = _create_variable(
                "post_2", [1, 1, self.skip_channels,
                           self.quantization_channels])

        return variables


    def _dilated_stack(self, X, dilation, layer_index):
        """create dilation layer or use it again.

        Parameters
        ==========
        X : np.ndarray or tf.tensor of shape(batch_size, height, width,
                                             in_channels)
            Input to the dilation stack
        dilation : int
            The dilation rate.
        layer_index : int
            Index of layer. Used for defining scope.

        Output
        ======
        residual, skip: np.ndarray of shape(batch_size, height, width,
                                            in_channels)
             Output of the dilated stack
        """
        with tf.variable_scope('dilated_layer_{}'.format(layer_index)):
            var_dict = self.variables['dilated_stack'][layer_index]
            conv_filter = _dilated_convolution(X, var_dict['filter'],
                                               dilation, name="conv_filter")
            conv_gate = _dilated_convolution(X, var_dict['gate'],
                                             dilation, name="conv_gate")

            # final output
            # Question: should the final skip and residual convolution have
            # different weight vector or same? here, the same is used.
            out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)
            out = tf.nn.conv2d(out, var_dict['skip'], padding="SAME", strides=[1,1,1,1])

            # return residual and skip output
            return out + X, out


    def create_network(self, X):
        """Create the network, by using dilated stack, postprocessing.

        Parameters
        ==========
        X : np.ndarray, of shape(batch, height, width, in_channels)
            The input data.
        
        Output
        ======
        conv2 : np.ndarray of shape(batch, height, width, in_channels)
            The output of the total network, unnormalized
        """
        with tf.variable_scope('initial_causal_conv'):
            initial_conv_result = tf.nn.conv2d(X, self.variables[
                                               'initial_filter'],
                                               padding="SAME", strides=[1,1,1,1])
        residual = initial_conv_result

        # create dilated stack results
        skip_list = list()
        with tf.variable_scope("dilated_stack"):
            for i, dilation in enumerate(self.dilations):
                residual, skip_result = self._dilated_stack(residual, dilation,
                                                            i)
                skip_list.append(skip_result)

        # post-processing
        # addition --> Relu --> convolution --> Relu --> convolution
        with tf.variable_scope("post_processing"):
            total_output = sum(skip_list)
            relu1 = tf.nn.tanh(total_output)
            conv1 = tf.nn.conv2d(relu1, self.variables['post_1'],
                                 padding="SAME", strides=[1,1,1,1])

            relu2 = tf.nn.tanh(conv1)
            conv2 = tf.nn.conv2d(relu2, self.variables['post_2'],
                                 padding="SAME", strides=[1,1,1,1])
        
        return conv2

    def loss(self, input_samples):
        """Generate the cross entropy loss and reduce mean between batches

        Parameters
        ==========
        input_samples : np.ndarray of shape(batch, height, width, in_channels)
             The input samples
        """
        with tf.variable_scope("loss"):
            # flip the input samples so that convolution depends on previous
            # samples
            input_samples =  tf.reverse(input_samples, [2])
            input_samples = _mu_law_encode(input_samples,
                                           self.quantization_channels)
            encoded = self._one_hot(input_samples)
            network_output = self.create_network(encoded)
            network_output = tf.reshape(network_output,
                                        [1, 1, -1,
                                         self.quantization_channels])
            

            # slice receptive field from the end(of flipped audio
            # signal) to preserve causility
            shape = network_output.shape
            receptive_samples = _get_rounded_receptive_samples(self.audio_frequency,
                                                       self.receptive_seconds)
            output_sliced = tf.slice(network_output, [0, 0, 0, 0],
                                      [-1, -1, int(shape[2]-receptive_samples),
                                       -1])
            encoded_sliced = tf.slice(encoded, [0, 0, 0, 0],
                                      [-1, -1, int(shape[2]-receptive_samples),
                                      -1])

            sliced_shape = encoded_sliced.shape
            # shift the input by left(reversed audio)
            encoded_shifted = tf.slice(tf.pad(encoded_sliced, [[0,0], [0,0], [1,0], [0,0]]),
                                       [0,0,0,0], [-1,-1, int(sliced_shape[2]),
                                                   -1])

            # reshape to find the cross entropy loss
            output_sliced = tf.reshape(output_sliced, [-1, self.quantization_channels])
            encoded_shifted = tf.reshape(encoded_shifted, [-1, self.quantization_channels])

            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits = output_sliced,
                labels = encoded_shifted)
            average_loss = tf.reduce_mean(loss)
            return average_loss

    def _generate_next_sample(self, waveform):
        """Generate the probabilty distribution of the next sample,
           based on current waveform.

        Parameters
        ==========
        waveform : np.ndarray of shape(batch, in_height, in_width,
                                       quantization_channels)
                    reversed input waveform

        Output
        ======
        new_waveform : np.ndarray of shape(batch, in_height,
                                           in_width,
                                           quantization_channels)
                    reversed generated waveform
        """
        with tf.variable_scope("Generate"):
            encoded = self._one_hot(waveform)
            network_output = self.create_network(encoded)
            out = tf.reshape(network_output, [-1, self.quantization_channels])
            prob = tf.nn.softmax(out)

            # return index + 1 to get the quantization channel value
            return tf.to_int32(tf.reshape(tf.argmax(prob, axis=1)[0], [1,1,1,1])) + 1

    def generate(self, seconds, song):
        """Generate audio based on trained model.

        Output
        ======
        generated_audio : np.ndarray of shape(out_width)
        """
        with tf.variable_scope("Generate"):
            receptive_samples = _get_rounded_receptive_samples(self.audio_frequency,
                                                       self.receptive_seconds)
            total_samples = _get_receptive_samples(self.audio_frequency,
                                                   seconds)
            
            # randomly generate first samples
            if len(song) < receptive_samples:
                print(len(song), receptive_samples)
                raise ValueError("enter longer song or shorter receptive field")
            current = song[1000:receptive_samples+3000]
            current = np.reshape(current, [1,1,current.shape[0], 1])
            total_waveform = tf.to_int32(tf.reverse(np.copy(current), [2]))
            current = tf.reverse(current, [2])
            current = _mu_law_encode(current, self.quantization_channels)

            for i in xrange(receptive_samples, total_samples):
                next_sample = self._generate_next_sample(current)
                total_waveform = tf.concat([next_sample, total_waveform], 2)

                # insert the next sample at the beginning and pop the last element
                current = tf.slice(current, [0,0,0,0], [-1,-1,int(current.shape[2]-1),-1])
                current = tf.concat([next_sample, current], 2)
                print(i)

            return _mu_law_decode(tf.reverse(total_waveform, [2]),
                                  self.quantization_channels)


    def _one_hot(self, input_samples):
        """Helper function to one_hot input samples.
        """
        encoded = tf.one_hot(input_samples, depth=self.quantization_channels,
                             dtype=tf.float32)
        return tf.reshape(encoded, [1, 1, -1, self.quantization_channels])
            

def _get_receptive_samples(audio_frequency, receptive_field):
    """helper function to get receptive seconds"""
    return audio_frequency * receptive_field


def _get_dilations(audio_frequency, receptive_field):
    """Create dilated factors list based on receiptive field
       These dilated factors are in the power of 2, till a max limit
       after which they start again.

    Parameters
    ==========
    audio_frequency : int, in Khz
        Frequency of the audio
    receptive_field : int,
        No of seconds to take into account
    """
    receptive_samples = _get_rounded_receptive_samples(audio_frequency,
                                               receptive_field)
    limit = np.log2(receptive_samples)
    dilated_list = list()
    counter = 0

    while True:
    
        for j in xrange(int(np.log2(MIN_DIL)), int(np.log2(MAX_DIL)) + 1):
            if counter == limit:
                return dilated_list
            dilated_list.append(2**j)
            counter += 1

def _get_rounded_receptive_samples(audio_frequency, receptive_field):
    """Get rounded receptive samples nearest to the power of 2
    """
    receptive_samples = _get_receptive_samples(audio_frequency,
                                               receptive_field)
    return  2 ** int(np.floor(np.log2(receptive_samples)))

def _mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    with tf.name_scope('encode'):
        mu = tf.to_float(quantization_channels - 1)
        # Perform mu-law companding transformation (ITU-T, 1988).
        # Minimum operation is here to deal with rare large amplitudes caused
        # by resampling.
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.to_int32((signal + 1) / 2 * mu + 0.5)

def _mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    # copied from https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/ops.py
    mu = quantization_channels - 1
    # Map values back to [-1, 1].
    signal = 2 * (tf.to_float(output) / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1. / mu) * ((1 + mu)**abs(signal) - 1)
    return tf.sign(signal) * magnitude
