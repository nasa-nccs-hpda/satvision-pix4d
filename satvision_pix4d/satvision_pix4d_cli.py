import os
import torch
import logging
import argparse

import warnings

warnings.filterwarnings("ignore", message=".*cuda capability 7.0.*")


from lightning.pytorch import Trainer

from satvision_pix4d.configs.config import _C, _update_config_from_file
from satvision_pix4d.utils import get_strategy, get_distributed_train_batches
from satvision_pix4d.pipelines import PIPELINES, get_available_pipelines
from satvision_pix4d.datamodules import DATAMODULES, get_available_datamodules


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main(config, output_dir):

    logging.info('Training')

    # Get the proper pipeline
    available_pipelines = get_available_pipelines()
    logging.info("Available pipelines:", available_pipelines)

    pipeline = PIPELINES[config.PIPELINE]
    logging.info(f'Using {pipeline}')

    ptlPipeline = pipeline(config)

    # Resume from checkpoint
    if config.MODEL.RESUME:
        logging.info(
            f'Attempting to resume from checkpoint {config.MODEL.RESUME}')
        ptlPipeline = pipeline.load_from_checkpoint(config.MODEL.RESUME)

    # Determine training strategy
    strategy = get_strategy(config)

    trainer = Trainer(
        accelerator=config.TRAIN.ACCELERATOR,
        devices=torch.cuda.device_count(),
        strategy=strategy,
        precision="32",#config.PRECISION,
        max_epochs=config.TRAIN.EPOCHS,
        log_every_n_steps=config.PRINT_FREQ,
        default_root_dir=output_dir,
    )

    # limit the number of train batches for debugging
    if config.TRAIN.LIMIT_TRAIN_BATCHES:
        trainer.limit_train_batches = get_distributed_train_batches(
            config, trainer)

    # setup datamodule
    if config.DATA.DATAMODULE:
        available_datamodules = get_available_datamodules()
        logging.info(f"Available data modules: {available_datamodules}")
        datamoduleClass = DATAMODULES[config.DATAMODULE]
        datamodule = datamoduleClass(config)
        logging.info(f'Training using datamodule: {config.DATAMODULE}')
        
        trainer.fit(model=ptlPipeline, datamodule=datamodule)

        # quick test of datamodule
        #datamodule.setup(stage=None)
        #print("Train dataset size:", len(datamodule.trainset))
        #print("Validation dataset size:", len(datamodule.validset))

        #sample = datamodule.trainset[0]
        #print("Sample type:", type(sample))

        # If your dataset returns a tuple
        #if isinstance(sample, tuple):
        #    x, y = sample
        #    print("x shape:", x.shape)
        #    print("y shape:", y.shape)

    else:
        logging.info(
            'Training without datamodule, assuming data is set' +
            f' in pipeline: {ptlPipeline}')
        trainer.fit(model=ptlPipeline)

    return


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c',
        '--config-path',
        type=str,
        required=True,
        help='Path to pretrained model config'
    )

    hparams = parser.parse_args()

    config = _C.clone()
    _update_config_from_file(config, hparams.config_path)

    output_dir = os.path.join(
        config.OUTPUT, config.MODEL.NAME, config.TAG)
    logging.info(f'Output directory: {output_dir}')
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(
        output_dir,
        f"{config.TAG}.config.json"
    )

    with open(path, "w") as f:
        f.write(config.dump())

    logging.info(f"Full config saved to {path}")
    logging.info(config.dump())

    # start main execution
    main(config, output_dir)

