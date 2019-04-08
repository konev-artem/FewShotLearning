from tensorflow.keras import layers, models, activations, backend
import tensorflow.keras as K
import fewshot.backbones.convnet as convnet


# Resnet-12 from "Dense Classsification and Implanting for Few-Shot Learning" Yann Lifchitz...
class ResidualBlock(layers.Layer):
    def __init__(self,  out_channels, activation='swish1', **kwargs):
        self.conv1 = convnet.ConvBlock(out_channels, activation, add_maxpool=False)
        self.conv2 = convnet.ConvBlock(out_channels, activation, add_maxpool=False)
        self.conv3 = convnet.ConvBlock(out_channels, activation, add_maxpool=True)
        
        self.conv_res  = layers.Conv2D(out_channels, 3, padding='same')
        self.bn_res = layers.BatchNormalization()
                
        self.out_channels = out_channels
        super(ResidualBlock, self).__init__(**kwargs)

    def call(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3.conv(y)
        y = self.conv3.bn(y)

        z = self.conv_res(x)
        z = self.bn_res(z)
        
        out = y + z
        out = self.conv3.nl(out)
        out = self.conv3.maxpool(out)
        
        return out

    def set_trainable(self, trainable):
        self.conv1.set_trainable(trainable)
        self.conv2.set_trainable(trainable)
        self.conv3.set_trainable(trainable)

        self.conv_res.trainable = trainable
        self.bn_res.trainable = trainable       
        
    def compute_output_shape(self, input_shape):
        return self.conv_res.compute_output_shape(input_shape)
    

class Resnet12:
    def __init__(self, input_size, activation='swish1'):
        self.residual1 = ResidualBlock(64, activation)
        self.residual2 = ResidualBlock(128, activation)
        self.residual3 = ResidualBlock(256, activation)
        self.residual4 = ResidualBlock(512, activation)
        self.inputs, self.outputs = self._build_net(input_size)
        
    def _build_net(self, input_size):
        input = layers.Input(shape=input_size)
        x = self.residual1(input)
        x = self.residual2(x)
        x = self.residual3(x)
        x = self.residual4(x)
        return [input], [x]

    def build_model(self):
        return models.Model(self.inputs, self.outputs)

    def set_trainable(self, trainable):
        self.residual1.set_trainable(trainable)
        self.residual2.set_trainable(trainable)
        self.residual3.set_trainable(trainable)
        self.residual4.set_trainable(trainable)

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs
