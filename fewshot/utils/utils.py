import tensorflow as tf


def join_models(base_model, head_layer_to_apply):
    outputs = head_layer_to_apply(*base_model.get_outputs())
    return tf.keras.models.Model(base_model.get_inputs(), outputs)


def reset_weights(model):
    session = tf.keras.backend.get_session()
    for weight in model.trainable_weights:
        weight.initializer.run(session=session)
