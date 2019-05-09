import os
import datetime

import tensorflow as tf

from ..special_layers import CosineLayer
from ...utils import join_models


def _train(backbone,
           head,
           loss,
           optimizer,
           train_generator,
           validation_generator=None,
           model_name='baseline',
           resume=None,
           save_checkpoints=True,
           checkpoint_dir='../fewshot/weights',
           save_best_only=True,
           period=10,
           tensorboard=False,
           log_dir='../fewshot/logs',
           reduce_lr=False,
           **kwargs):

    model = join_models(backbone, head)
    model.compile(optimizer, loss, metrics=["accuracy"])

    if resume is not None:
        model.load_weights(resume)

    callbacks = [
        tf.keras.callbacks.TerminateOnNaN()
    ]

    if save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)
        filepath = os.path.join(checkpoint_dir,
                                model_name + ".{epoch:02d}-{loss:.2f}.hdf5")
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(filepath,
                                               monitor="loss",
                                               save_best_only=save_best_only,
                                               period=period))

    if tensorboard:
        log_dir = os.path.join(log_dir,
            '{}_{}'.format(model_name, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
        os.makedirs(log_dir, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=log_dir))

    if reduce_lr:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau())

    model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        callbacks=callbacks,
        **kwargs
    )

    return backbone


def simple_one_layer_cross_entropy_train(backbone,
                                         train_dataset,
                                         validation_dataset=None,
                                         **kwargs):
    return _train(
        backbone=backbone,
        head=tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(train_dataset.n_classes),
        ]),
        loss=lambda y_true, y_pred: tf.keras.losses.categorical_crossentropy(
            y_true, y_pred, from_logits=True),
        train_generator=train_dataset,
        validation_generator=validation_dataset,
        **kwargs
    )


def simple_cosine_layer_cross_entropy_train(backbone,
                                         train_dataset,
                                         validation_dataset=None,
                                         **kwargs):
    return _train(
        backbone=backbone,
        head=tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            CosineLayer(train_dataset.n_classes),
        ]),
        loss=lambda y_true, y_pred: tf.keras.losses.categorical_crossentropy(
            y_true, y_pred, from_logits=True),
        train_generator=train_dataset,
        validation_generator=validation_dataset,
        **kwargs
    )
