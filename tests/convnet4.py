import unittest

import numpy as np
from tensorflow.keras import models
from tensorflow.keras import backend as K

from basenets.convnet import ConvNet, ConvBlock


class TestConvnet(unittest.TestCase):

    def test_forward(self):
        net = ConvNet()
        model = models.Model(*net.build_net((256, 256, 3)))

        input = np.random.random((2, 256, 256, 3)).astype(np.float32)
        self.assertEqual((2, 16, 16, 64), model(input).shape)

    def test_fit(self):
        net = ConvNet()
        model = models.Model(*net.build_net((256, 256, 3)))
        model.compile(loss='mean_squared_error', optimizer='sgd')

        input = np.random.random((2, 256, 256, 3)).astype(np.float32)
        target = np.random.random((2, 16, 16, 64)).astype(np.float32)

        convs = [block.conv for block in net.blocks]

        conv_weights = self.get_conv_weights(convs)
        model.fit(input, target)
        new_conv_weights = self.get_conv_weights(convs)

        for source, target in zip(conv_weights, new_conv_weights):
            self.assertFalse(np.allclose(source, target))

    def test_freeze(self):
        net = ConvNet()
        model = models.Model(*net.build_net((256, 256, 3)))
        net.set_trainable(False)
        model.compile(loss='mean_squared_error', optimizer='sgd')

        input = np.random.random((2, 256, 256, 3)).astype(np.float32)
        target = np.random.random((2, 16, 16, 64)).astype(np.float32)

        convs = [block.conv for block in net.blocks]

        conv_weights = self.get_conv_weights(convs)
        model.fit(input, target)
        new_conv_weights = self.get_conv_weights(convs)

        for source, target in zip(conv_weights, new_conv_weights):
            self.assertTrue(np.allclose(source, target))

    def test_compute_conv_block_output_shape(self):
        conv = ConvBlock(512)
        actual = conv.compute_output_shape((2, 256, 256, 3))
        self.assertTrue((2, 128, 128, 512), actual)

        conv = ConvBlock(512, False)
        actual = conv.compute_output_shape((2, 256, 256, 3))
        self.assertTrue((2, 256, 256, 3), actual)

    def get_conv_weights(self, convs):
        r = []
        for conv in convs:
            r.append(K.eval(conv.weights[0]))
        return r
