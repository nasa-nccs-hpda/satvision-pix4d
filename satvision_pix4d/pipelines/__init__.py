from satvision_pix4d.pipelines.satvision_pix4d_pretrain import SatVisionPix4DSatMAEPretrain


PIPELINES = {
    'satvision_pix4d_satmae_pretrain': SatVisionPix4DSatMAEPretrain,
}


def get_available_pipelines():
    return {name: cls for name, cls in PIPELINES.items()}
