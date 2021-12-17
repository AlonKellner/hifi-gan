def parse_layer_params(layer_params_config):
    layer_params = layer_params_config.split(',')
    layer_params = [(int(param) if str.isdigit(param) else param) for param in layer_params]
    return layer_params


def parse_layer(layer_config: str):
    layer_types_config, layer_params_config = layer_config.split(':')
    layer_types = parse_layer_params(layer_types_config)
    layer_params = parse_layer_params(layer_params_config)
    return layer_types, layer_params


def parse_layers(layers_config: list):
    return [parse_layer(layer_config) for layer_config in layers_config]