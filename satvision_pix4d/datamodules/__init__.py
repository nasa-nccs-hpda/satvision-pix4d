from satvision_pix4d.datamodules.abi_temporal_datamodule \
    import ABITemporalDataModule
from satvision_pix4d.datamodules.abi_temporal_benchmark_datamodule \
    import ABITemporalBenchmarkDataModule

DATAMODULES = {
    'abi_temporal': ABITemporalDataModule,
    'abi_temporal_benchmark': ABITemporalBenchmarkDataModule
}


def get_available_datamodules():
    return {name: cls for name, cls in DATAMODULES.items()}
