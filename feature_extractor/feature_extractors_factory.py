import tensorflow as tf
import tensorflow.keras.layers as L


from .feature_extractor import FeatureExtractor


class ConventionalFourConv(FeatureExtractor):
    def _conv_block(self, input):
        output = L.Conv2D(64, 3)(input)
        output = L.BatchNormalization()(output)
        output = L.ReLU()(output)
        output = L.MaxPool2D(2)(output)
        return output

    def __init__(self, image_size):
        with tf.scope("4conv"):
            self.input = tf.placeholder(tf.float32, shape=(None, *image_size, 3))
            self.output = self.input
            for i in range(4):
                self.output = self._conv_block(self.output)

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output