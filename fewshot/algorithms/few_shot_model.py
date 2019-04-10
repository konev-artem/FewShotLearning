import tensorflow as tf

from ..utils import join_models


class CosineLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes):
        super(CosineLayer, self).__init__()
        self.num_classes = num_classes
        self.W = tf.Variable(self._W_init_value(), trainable=True)

    def call(self, input, **kwargs):
        dot_product = tf.matmul(input, self.W, transpose_b=True)
        f_norm = tf.pow(tf.reduce_sum(tf.multiply(input, input), axis=1, keepdims=True), 0.5)
        w_norm = tf.pow(tf.reduce_sum(tf.multiply(self.W, self.W), axis=0, keepdims=True), 0.5)

        return dot_product / f_norm / w_norm

    def reset(self):
        self.W.assign(self._W_init_value())

    def _W_init_value(self):
        # change it
        return tf.zeros(input.get_shape()[1], self.num_classes)


class BaselineFewShotModel(BaseFewShotModel):
    def __init__(self, backbone, num_classes):
        self.head_layer = CosineLayer(num_classes)
        self.backbone = backbone
        self.backbone.set_trainable(False)
        output = self.head_layer(backbone.get_outputs[0])
        self.model = tf.keras.models.Model(self.backbone.get_inputs(), output)

    def fit(self, episode_dataset, batch_size=32, iters=1000, optimizer="adam"):
        self.reset(optimizer)
        self.model.fit_generator(episode_dataset.get_batch_generator(
            batch_size=batch_size), steps_per_epoch=iters, epochs=1)

    def predict(self, episode_dataset):
        return self.model.predict_generator(episode_dataset) # really?

    def reset(self, optimizer):
        self.head_layer.reset()
        self.model.compile(optimizer) # probably we need to completely destroy to prevent memory leaks
