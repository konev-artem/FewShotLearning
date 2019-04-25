import tensorflow as tf

from fewshot.algorithms.fewshot_models import FewShotModelBase
from fewshot.algorithms.special_layers import CosineLayer
from fewshot.utils import reset_weights


class BaselineFewShotModel(FewShotModelBase):
    def __init__(self, backbone, num_classes):
        self.head_layer = tf.keras.Sequential(layers=[tf.keras.layers.Flatten(), CosineLayer(num_classes)])
        self.backbone = backbone
        self.backbone.set_trainable(False)
        output = self.head_layer(backbone.get_outputs()[0])
        self.model = tf.keras.models.Model(self.backbone.get_inputs(), output)

    def fit(self, x, y, batch_size, epochs, optimizer, **kwargs):
        self.reset(optimizer)
        self.model.fit(x, y, batch_size, epochs, **kwargs)

    def fit_generator(self, episode_generator, n_epochs=10, optimizer="adam"):
        self.reset(optimizer)
        self.model.fit_generator(episode_generator, epochs=n_epochs, workers=0, verbose=0)

    def predict(self, x, batch_size, **kwargs):
        return self.model.predict(x, batch_size, **kwargs)

    def predict_generator(self, episode_dataset):
        return self.model.predict_generator(episode_dataset, workers=0) # really?

    def reset(self, optimizer):
        reset_weights(self.head_layer)
        self.model.compile(
            optimizer,
            lambda y_true, y_pred: tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True),
            metrics=["accuracy"]
        )  # probably we need to completely destroy to prevent memory leaks
