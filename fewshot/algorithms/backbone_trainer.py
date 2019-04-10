from tensorflow.python.keras import layers, losses

from ..utils import join_models

#TODO: Remove this, when the issue with dataframe is fixed
def inf_generator(generator):
    while True:
        yield generator.next()

def _train(backbone, heads, loss, optimizer, n_epochs, batch_size, train_dataset, validation_dataset, **kwargs):
    model = join_models(backbone, *heads)
    model.compile(optimizer, loss, metrics="accuracy")

    #TODO: Fix this
    train_generator = train_dataset.get_batch_generator(batch_size=batch_size)
    validation_generator = validation_dataset.get_batch_generator(batch_size=batch_size)

    model.fit_generator(
        inf_generator(train_generator),
        steps_per_epoch=len(train_generator),
        epochs=n_epochs,
        validation_data=inf_generator(validation_generator),
        validation_steps=len(validation_generator),
    )
    return backbone


def simple_one_layer_cross_entropy_train(
        backbone, train_dataset, validation_dataset=None, n_epochs=1, batch_size=32, optimizer="adam"):
    return _train(
        backbone=backbone,
        heads=[layers.Flatten(), layers.Dense(train_dataset.n_classes)],
        loss=lambda y_true, y_pred: losses.categorical_crossentropy(y_true, y_pred, from_logits=True),
        optimizer=optimizer,
        n_epochs=n_epochs,
        batch_size=batch_size,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset
    )



