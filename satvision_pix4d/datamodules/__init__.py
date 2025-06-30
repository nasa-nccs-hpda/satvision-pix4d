from satvision_pix4d.datamodules.abi_temporal_datamodule import ABITemporalDataModule


DATAMODULES = {
    'abi_temporal': ABITemporalDataModule,
}


def get_available_datamodules():
    return {name: cls for name, cls in DATAMODULES.items()}
