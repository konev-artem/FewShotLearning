import tensorflow as tf

from fewshot.utils import join_models


def _train(backbone, head, loss, optimizer, n_epochs,
           train_generator, validation_generator, checkpoint, callbacks):
    model = join_models(backbone, head)
    model.compile(optimizer, loss, metrics=["accuracy"])
    if checkpoint:
        model.load_weights(checkpoint)

    model.fit_generator(
        train_generator,
        epochs=n_epochs,
        validation_data=validation_generator,
        use_multiprocessing=False,
        workers=0,  # TODO: Override __getitem__ method in fewshot.data_provider.generator.DataFrameIterator
        callbacks=callbacks
    )

    return backbone


def simple_one_layer_cross_entropy_train(backbone, train_dataset, validation_dataset=None,
                                         n_epochs=1, optimizer="adam", checkpoint=None,
                                         callbacks=[]):
    return _train(
        backbone=backbone,
        head=tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(train_dataset.n_classes),
        ]),
        loss=lambda y_true, y_pred: tf.keras.losses.categorical_crossentropy(
            y_true, y_pred, from_logits=True),
        optimizer=optimizer,
        n_epochs=n_epochs,
        train_generator=train_dataset,
        validation_generator=validation_dataset,
        checkpoint=checkpoint,
        callbacks=callbacks
    )
