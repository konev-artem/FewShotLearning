import tensorflow as tf

from fewshot.algorithms.fewshot_models import FewShotModelBase
from fewshot.algorithms.special_layers import CosineLayer
from fewshot.utils import reset_weights


class BaselineFewShotModel(FewShotModelBase):
    def __init__(self, backbone, num_classes, with_cosine=True):
        if with_cosine:
            logits_layer = CosineLayer(num_classes)
        else:
            logits_layer = tf.keras.layers.Dense(num_classes)

        self.head_layer = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            logits_layer
        ])

        self.backbone = backbone
        self.backbone.set_trainable(False)
        output = self.head_layer(backbone.get_outputs()[0])
        self.model = tf.keras.models.Model(self.backbone.get_inputs(), output)
        self.loss = lambda y_true, y_pred: \
            tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
        self.metrics = ["accuracy"]

    def fit(self, x, y, optimizer, **kwargs):
        self.reset(optimizer)
        self.model.fit(x, y, **kwargs)

    def fit_generator(self, generator, optimizer, **kwargs):
        self.reset(optimizer)
        self.model.fit_generator(generator, **kwargs)

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)

    def predict_generator(self, dataset, **kwargs):
        return self.model.predict_generator(dataset, **kwargs)

    def recompile(self, optimizer):
        # probably we need to completely destroy to prevent memory leaks
        self.model.compile(optimizer, self.loss, metrics=self.metrics)

    def reset(self, optimizer):
        reset_weights(self.head_layer)
        self.recompile(optimizer)
