import tensorflow as tf
import enum

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

_configurations = {
    'A': [[64], [128], [256,256], [512,512], [512,512]],
    'D': [[64,64], [128,128], [256,256,256], [512,512,512], [512,512,512]],
    'E': [[64,64], [128,128], [256,256,256,256], [512,512,512,512], [512,512,512,512]],
}
_configurations['11'] = _configurations['A']
_configurations['16'] = _configurations['D']
_configurations['19'] = _configurations['E']


class Model(object):
    def __init__(self, vgg_size, n_outputs, data_format, use_bn):
        split_vgg_params = vgg_size.split('_')
        self.vgg_size = split_vgg_params[0]
        self.no_fc = False
        self.use_bn = use_bn

        for param in split_vgg_params[1:]:
            negate = param[:2] == 'no'
            arg = param[-2:]
            if arg == 'fc':
                self.no_fc = negate
            elif arg == 'bn':
                self.use_bn = not negate
            else:
                raise ValueError(param)

        self.n_outputs = n_outputs

        if not data_format:
            data_format = ('channels_first' if tf.test.is_built_with_cuda()
                           else 'channels_last')
        if data_format == 'channels_first':
            data_format = 'NCHW'
        elif data_format == 'channels_last':
            data_format = 'NHWC'
        else:
            raise ValueError(data_format)
        self.data_format = data_format


    def __call__(self, inputs, is_training):
        self.is_training = is_training
        self.layer_idx = 1


        if self.data_format == 'NCHW':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        for block in _configurations[self.vgg_size]:
            inputs = self.block(inputs, block)

        if self.data_format == 'NHWC':
            size = inputs.shape[1] * inputs.shape[2]
        else:
            size = inputs.shape[2] * inputs.shape[3]

        outputs = tf.reshape(inputs, (-1, 512*size))
        if not self.no_fc:
            outputs = self.fc(outputs, 4096)
            outputs = self.fc(outputs, 4096)
        outputs = self.fc(outputs, self.n_outputs, nonlinear=False)
        return outputs

    def block(self, inputs, layers):
        for filters in layers:
            inputs = self.conv2d(inputs, filters)

        if self.data_format == 'NHWC':
            kernel = [1,2,2,1]
        elif self.data_format == 'NCHW':
            kernel = [1,1,2,2]
        else:
            raise ValueError('Unsupported data format {}'.format(self.data_format))

        return tf.nn.max_pool(
            inputs,
            kernel,
            kernel,
            padding='VALID',
            data_format=self.data_format,
        )

    def fc(self, inputs, to_, nonlinear=True):
        from_ = inputs.shape[1]
        fc_w = self.get_weight((from_, to_), inputs.dtype, False)
        fc_m = self.get_mask((from_, to_), inputs.dtype, False)
        fc_b = self.get_bias((1, to_), inputs.dtype, False)
        self.layer_idx += 1

        output = inputs @ (fc_w * fc_m) + fc_b
        if nonlinear:
            output = tf.nn.relu(output)
            output = tf.nn.dropout(output, rate=0.5)
        return output

    def conv2d(self, inputs, filters):
        if self.data_format == 'NCHW':
            input_dim = inputs.shape[1]
        else:
            input_dim = inputs.shape[-1]

        kernel_shape = (3, 3, input_dim, filters)

        kernel = self.get_weight(kernel_shape, inputs.dtype, True)
        mask = self.get_mask(kernel_shape, inputs.dtype, True)
        self.layer_idx += 1
        filters = tf.math.multiply(kernel, mask)

        outputs = tf.nn.conv2d(
            inputs, filters, [1,1,1,1],
            padding='SAME',
            data_format=self.data_format,
        )

        if self.use_bn:
            outputs = tf.layers.batch_normalization(
                inputs=outputs, axis=1 if self.data_format == 'NCHW' else 3,
                momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
                scale=True, training=self.is_training, fused=True)

        return tf.nn.relu(outputs)

    def get_weight(self, shape, dtype, is_conv):
        if is_conv:
            name = 'conv'
            initializer = tf.compat.v1.variance_scaling_initializer()
        else:
            name = 'fc'
            initializer = tf.random_normal_initializer(stddev=.01)

        return tf.get_variable(
            initializer=initializer,
            trainable=True,
            shape=shape,
            dtype=dtype,
            name='{}_{}_w'.format(name, self.layer_idx),
        )

    def get_bias(self, shape, dtype, is_conv):
        if is_conv:
            name = 'conv'
        else:
            name = 'fc'

        return tf.get_variable(
            initializer=tf.zeros_initializer(),
            trainable=True,
            shape=shape,
            dtype=dtype,
            name='{}_{}_b'.format(name, self.layer_idx),
        )

    def get_mask(self, shape, dtype, is_conv):
        if is_conv:
            name = 'conv'
        else:
            name = 'fc'

        return tf.get_variable(
            initializer=tf.ones_initializer(),
            trainable=False,
            shape=shape,
            dtype=dtype,
            name='{}_{}_m'.format(name, self.layer_idx),
        )
