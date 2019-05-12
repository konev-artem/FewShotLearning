import tensorflow as tf
from tensorflow.python.keras import initializers

def pairwise_cosine(X, Y, transpose_Y=True):
    if transpose_Y:
        Y = tf.transpose(Y)

    dot_product = tf.matmul(X, Y)
    x_norm = tf.pow(tf.reduce_sum(tf.multiply(X, X), axis=1, keepdims=True) + 1e-7, 0.5)
#     y_norm = tf.pow(tf.reduce_sum(tf.multiply(Y, Y), axis=0, keepdims=True) + 1e-7, 0.5)

    return dot_product / x_norm


class CosineLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes, scale_factor=10):
        self.num_classes = num_classes
        self.scale_factor = scale_factor
        super(CosineLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 2, input_shape

        self.v = self.add_weight(
            name='kernel',
            shape=(input_shape[1].value, self.num_classes),
            initializer='uniform',
            trainable=True
        )
        self.g = self.add_weight(
                  name="g",
                  shape=(1,self.num_classes),
                  initializer=initializers.get('ones'),
                  dtype= tf.float32,
                  trainable=True)

        super(CosineLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[1].value, self.num_classes)

    def call(self, input, **kwargs):
        return self.scale_factor * pairwise_cosine(input, self.g * self.v / tf.norm(self.v, axis=-2, keepdims=True), transpose_Y=False)
