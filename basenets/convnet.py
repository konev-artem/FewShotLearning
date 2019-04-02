from tensorflow.keras import layers, models, activations, backend


class ConvBlock(layers.Layer):
    def __init__(self,  out_channels, add_maxpool = True, **kwargs):
        self.conv = layers.Conv2D(out_channels, 3, padding='same')
        self.bn = layers.BatchNormalization()
        self.nl = layers.ReLU()
        
        if add_maxpool:
            self.maxpool = layers.MaxPool2D()
        else:
            self.maxpool = None
        
        self.out_channels = out_channels
        super(ConvBlock, self).__init__(**kwargs)

    def call(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.nl(out)
        
        if self.maxpool is not None:
            out = self.maxpool(out)
        
        return out

    def set_trainable(self, trainable):
        self.conv.trainable = trainable
        self.bn.trainable = trainable

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_channels)


class ConvNet:
    def __init__(self, depth=4):
        self.blocks = []
        for i in range(depth):
            outdim = 64
            self.blocks.append(ConvBlock(outdim, i < 4))
                    
    def build_net(self, input_size):
        input = layers.Input(shape=input_size)
        x = input
        for block in self.blocks:
            x = block(x)
            
        return [input], [x]
    
    def set_trainable(self, trainable):
        for block in self.blocks:
            block.set_trainable(trainable)