from monai.networks.nets import SwinUNETR


def create_model():

    model = SwinUNETR(
        in_channels=4,
        out_channels=4,
        feature_size=24,
        spatial_dims=3,
        use_checkpoint=False,
    )

    return model