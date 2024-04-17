from monai.networks.nets import DynUNet, UNet

def get_model(cfg):
    _, model, n_filters, depth, num_res_units = cfg.run.arch.split('-')
    n_filters, depth, num_res_units = int(n_filters), int(depth), int(num_res_units)
    model_config = cfg[model][cfg.run.dataset_key]
    net = None
    if model == 'unet':
        channels = [n_filters * 2 ** i for i in range(depth)]
        strides = [2] * (depth - 1)
        net = UNet(
            spatial_dims=2,
            in_channels=model_config.n_chans_in,
            out_channels=model_config.n_chans_out,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units
        )
    elif model == 'DynUNet':
        kernels = [[3,3],[3,3],[3,3],[3,3],[3,3],[3,3],[3,3]]
        strides = [[1,1],[2,2],[2,2],[2,2],[2,2],[2,2],[2,2]]
        net = DynUNet(
            spatial_dims=2,
            in_channels=model_config.n_chans_in,
            out_channels=model_config.n_chans_out,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:]
        )
    assert net is not None, 'Invalid model argument'
    return net

class ModelManager:
    pass
