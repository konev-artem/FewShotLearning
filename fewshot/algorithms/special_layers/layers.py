import tensorflow as tf


class CosineLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(CosineLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 2, input_shape

        self.W = self.add_weight(
            name='kernel',
            shape=(input_shape[1].value, self.num_classes),
            initializer='uniform',
            trainable=True
        )

        super(CosineLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[1].value, self.num_classes)

    def call(self, input, **kwargs):
        dot_product = tf.matmul(input, self.W)
        f_norm = tf.pow(tf.reduce_sum(tf.multiply(input, input), axis=1, keepdims=True), 0.5)
        w_norm = tf.pow(tf.reduce_sum(tf.multiply(self.W, self.W), axis=0, keepdims=True), 0.5)

        return dot_product / f_norm / w_norm
