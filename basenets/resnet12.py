from tensorflow.keras import layers, models, activations, backend
import tensorflow.keras as K


# swish block from  "Searching for activation functions" P. Ramachandran, B. Zoph, and Q. V. Le.
def swish1(x):
    return x * activations.sigmoid(x)


# Resnet-12 from "Dense Classsification and Implanting for Few-Shot Learning" Yann Lifchitz...
class ResidualBlock(layers.Layer):
    def __init__(self,  out_channels, **kwargs):
        self.conv1 = layers.Conv2D(out_channels, 3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.nl1 = layers.Lambda(swish1)
        
        self.conv2 = layers.Conv2D(out_channels, 3, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.nl2 = layers.Lambda(swish1)
        
        self.conv3 = layers.Conv2D(out_channels, 3, padding='same')
        self.bn3 = layers.BatchNormalization()
        self.nl3 = layers.Lambda(swish1)
        
        self.conv_res  = layers.Conv2D(out_channels, 3, padding='same')
        self.bn_res = layers.BatchNormalization()
                
        self.out_channels = out_channels
        super(ResidualBlock, self).__init__(**kwargs)

    def call(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.nl1(y)
               
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.nl2(y)
        
        y = self.conv3(y)
        y = self.bn3(y)
        
        z = self.conv_res(x)
        z = self.bn_res(z)
        
        out = y + z
        out = self.nl3(out)
        
        return out

    def set_trainable(self, trainable):
        self.conv1.trainable = trainable
        self.bn1.trainable = trainable

        self.conv2.trainable = trainable
        self.bn2.trainable = trainable
        
        self.conv3.trainable = trainable
        self.bn3.trainable = trainable
        
        self.conv_res.trainable = trainable
        self.bn_res.trainable = trainable       
        
    def compute_output_shape(self, input_shape):
        return self.conv_res.compute_output_shape(input_shape)
    

class Resnet12:
    def __init__(self):
        self.residual1 = ResidualBlock(64)
        self.max_pool1 = layers.MaxPool2D(padding='same')
        self.residual2 = ResidualBlock(128)
        self.max_pool2 = layers.MaxPool2D(padding='same')
        self.residual3 = ResidualBlock(256)
        self.max_pool3 = layers.MaxPool2D(padding='same')
        self.residual4 = ResidualBlock(512)
        self.max_pool4 = layers.MaxPool2D(padding='same')
        
        
    def build_net(self, input_size):
        input = layers.Input(shape=input_size)
        x = self.residual1(input)
        x = self.max_pool1(x)
        x = self.residual2(x)
        x = self.max_pool2(x)
        x = self.residual3(x)
        x = self.max_pool3(x)
        x = self.residual4(x)
        x = self.max_pool4(x)       
        return [input], [x]
    
    def set_trainable(self, trainable):
        self.residual1.set_trainable(trainable)
        self.residual2.set_trainable(trainable)
        self.residual3.set_trainable(trainable)
        self.residual4.set_trainable(trainable)
