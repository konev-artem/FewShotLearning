import tensorflow as tf

from fewshot.utils import join_models


def build_one_layer_classifier(backbone, n_classes):
    head = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(n_classes)
    ])

    model = join_models(backbone, head)
    return model


def _train(model, loss, optimizer, n_epochs, train_generator, validation_generator):
    model.compile(optimizer, loss, metrics=["accuracy"])
    model.fit_generator(
        train_generator,
        epochs=n_epochs,
        validation_data=validation_generator,
        use_multiprocessing=False,
        workers=0  # TODO: Override __getitem__ method in fewshot.data_provider.generator.DataFrameIterator
    )
    return model  # TODO: should we return classifier or just backbone?


def cross_entropy_train(
        backbone_classifier, train_dataset, validation_dataset=None, n_epochs=1, optimizer="adam"):
    return _train(
        model=backbone_classifier,
        loss=lambda y_true, y_pred: tf.keras.losses.categorical_crossentropy(
            y_true, y_pred, from_logits=True),
        optimizer=optimizer,
        n_epochs=n_epochs,
        train_generator=train_dataset,
        validation_generator=validation_dataset
    )


def simple_one_layer_cross_entropy_train(
        backbone, train_dataset, validation_dataset=None, n_epochs=1, optimizer="adam"):
    backbone_classifier = build_one_layer_classifier(backbone, train_dataset.n_classes)
    return cross_entropy_train(backbone_classifier, train_dataset, validation_dataset,
                               n_epochs, optimizer)
