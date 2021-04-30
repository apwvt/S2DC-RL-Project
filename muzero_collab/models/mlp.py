import torch

def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
):
    """
    Utility to create a fully-connected network from config
    while guaranteeing reasonable input/output sizes.
    """
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []

    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]

    return torch.nn.Sequential(*layers)
