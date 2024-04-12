def create(dim, architecture,
           backbone, kernel_size, block_filters, block_per_level, normalization, depth,
           outputs, output_activation, name=None):
    # import inside function to avoid import tensorflow
    import models.architecture as a
    import models.backbone as b

    if type(dim) is str:
        if dim.lower() == '2d':
            dim = 2
        elif dim.lower() == '3d':
            dim = 3
    if name is None:
        name = f'{architecture}-{dim}D-{block_filters}-{depth}'.upper()
    assert 2 <= dim <= 3
    assert architecture in ['u-net']
    assert backbone in ['vgg', 'residual']
    assert 1 <= kernel_size <= 32
    assert 1 <= block_filters <= 512
    assert 1 <= block_per_level <= 16
    assert normalization is False or normalization == 'batchnorm'
    assert 1 <= depth <= 16
    assert outputs >= 1
    assert output_activation in ['sigmoid', 'tanh', 'linear', 'softmax']

    if dim == 2:
        input_shape = (None, None, 1)
    elif dim == 3:
        input_shape = (None, None, None, 1)
    else:
        raise NotImplementedError

    if architecture == 'u-net':
        architecture = a.UNet(input_shape=input_shape,
                              depth=depth,
                              output_classes=outputs,
                              output_activation=output_activation,
                              op_dim=dim,
                              dropout=0,
                              pool_size=2,
                              multiple_outputs=False,
                              name=name)
    else:
        raise NotImplementedError
    if backbone in ['vgg', 'residual']:
        model_backbone = b.VGG(initial_block_depth=block_filters,
                               initial_block_length=block_per_level,
                               activation='relu',
                               kernel_size=kernel_size,
                               normalization=normalization)
        if backbone == 'residual':
            model_backbone = b.ResBlock(model_backbone)
    else:
        raise NotImplementedError

    return architecture(model_backbone, model_backbone)
