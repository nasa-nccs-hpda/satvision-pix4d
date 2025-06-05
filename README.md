# satvision-pix4d

SatVision PIX4D

## Download Container

```bash
module load singularity
singularity build --sandbox /lscratch/jacaraba/container/satvision-pix4d docker://nasanccs/satvision-pix4d:latest
```

## Random Tiles Generator

```bash
singularity exec --env PYTHONPATH=/explore/nobackup/people/jacaraba/development/satvision-pix4d --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects /lscratch/jacaraba/container/satvision-pix4d python /explore/nobackup/people/jacaraba/development/satvision-pix4d/satvision_pix4d/view/abi_tiles_generator_pipeline_cli.py
```

### Gathering Some Metrics

- Just to stack the 16 bands for a single time period ~40GB max of RAM, 3 minutes 30 seconds (need 7 timesteps)

## Stratified Tiles Generator

### Bucket #1: Convection Tiles

```bash
singularity exec --env PYTHONPATH=/explore/nobackup/people/jacaraba/development/satvision-pix4d --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects /lscratch/jacaraba/container/satvision-pix4d python /explore/nobackup/people/jacaraba/development/satvision-pix4d/satvision_pix4d/view/abi_tiles_generator_pipeline_cli.py
```

### Bucket #2: Cloud Feature Tiles

### Bucket #3: Land Cover Tiles
