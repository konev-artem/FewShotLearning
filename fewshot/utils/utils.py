from tensorflow.keras import models


def join_models(base_model, *head_layers_to_apply):
    outputs = base_model.get_outputs()
    print(type(head_layers_to_apply))
    for layer in head_layers_to_apply:
        if isinstance(outputs, list):
            outputs = layer(*outputs)
        else:
            outputs = layer(outputs)
    return models.Model(base_model.get_inputs(), outputs)
