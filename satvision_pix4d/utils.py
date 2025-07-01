import os
import logging
import argparse
from datetime import datetime


# -----------------------------------------------------------------------------
# get_strategy
# -----------------------------------------------------------------------------
def get_strategy(config):

    strategy = config.TRAIN.STRATEGY

    if strategy == 'deepspeed':
        deepspeed_config = {
            "train_micro_batch_size_per_gpu": config.DATA.BATCH_SIZE,
            "steps_per_print": config.PRINT_FREQ,
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {
                "stage": config.DEEPSPEED.STAGE,
                "contiguous_gradients":
                    config.DEEPSPEED.CONTIGUOUS_GRADIENTS,
                "overlap_comm": config.DEEPSPEED.OVERLAP_COMM,
                "reduce_bucket_size": config.DEEPSPEED.REDUCE_BUCKET_SIZE,
                "allgather_bucket_size":
                    config.DEEPSPEED.ALLGATHER_BUCKET_SIZE,
            },
            "activation_checkpointing": {
                "partition_activations": config.TRAIN.USE_CHECKPOINT,
            },
        }

        """
        TODO: this setting is being set somewhere in the pipeline
        and I was not able to rewrite it. We will check on this later,
        I just do not have enough time right now. Leaving some code here
        to document this attempt.
        """

        # Check if the user specified a cache dir
        if config.TRAIN.TRITON_CACHE_DIR is not None:

            triton_cache_dir = config.TRAIN.TRITON_CACHE_DIR

        elif os.path.exists("/lscratch"):

            # Get username
            user = os.environ.get("USER", "default_user")

            # Construct path
            triton_cache_dir = f"/lscratch/{user}/triton_cache"

        else:
            logging.info("TRITON_CACHE_DIR using default location.")
            from lightning.pytorch.strategies import DeepSpeedStrategy
            return DeepSpeedStrategy(config=deepspeed_config)

        # For any other path, just create and use it
        os.makedirs(triton_cache_dir, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = triton_cache_dir
        logging.info(f"Using TRITON_CACHE_DIR = {triton_cache_dir}")

        from lightning.pytorch.strategies import DeepSpeedStrategy
        return DeepSpeedStrategy(config=deepspeed_config)

    else:
        # These may be return as strings
        return strategy


# -----------------------------------------------------------------------------
# get_distributed_train_batches
# -----------------------------------------------------------------------------
def get_distributed_train_batches(config, trainer):
    if config.TRAIN.NUM_TRAIN_BATCHES:
        return config.TRAIN.NUM_TRAIN_BATCHES
    else:
        return config.DATA.LENGTH // \
            (config.DATA.BATCH_SIZE * trainer.world_size)


# -------------------------------------------------------------------------
# validate_date
# -------------------------------------------------------------------------
def valid_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "not a valid date: {0!r}".format(s)
        raise argparse.ArgumentTypeError(msg)
