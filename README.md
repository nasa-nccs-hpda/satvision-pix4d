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

## Convection Tiles Generator

```bash
singularity exec --env PYTHONPATH=/explore/nobackup/people/jacaraba/development/satvision-pix4d --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects,/css /lscratch/jacaraba/container/satvision-pix4d python /explore/nobackup/people/jacaraba/development/satvision-pix4d/satvision_pix4d/view/abi_tiles_generator_pipeline_cli.py --stratification convection --convection-regex "/explore/nobackup/projects/pix4dcloud/Jingbo/cloudsystem_mask_2019-2020/2020*.nc" --output-dir /explore/nobackup/projects/pix4dcloud/jacaraba/tiles_pix4d
```

with local files (a little bit broken for now, some files were never downloaded)

```bash
singularity exec --env PYTHONPATH=/explore/nobackup/people/jacaraba/development/satvision-pix4d --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects,/css,/nfs4m /lscratch/jacaraba/container/satvision-pix4d python /explore/nobackup/people/jacaraba/development/satvision-pix4d/satvision_pix4d/view/abi_tiles_generator_pipeline_cli.py --stratification convection --convection-regex "/explore/nobackup/projects/pix4dcloud/Jingbo/cloudsystem_mask_2019-2020/2020*.nc" --output-dir /explore/nobackup/projects/pix4dcloud/jacaraba/tiles_pix4d --tile-size 512 --channels 1 2 --local-data-dir '/css/geostationary/BackStage/GOES-16-ABI-L1B-FULLD'
```

from AWS only

```bash
singularity exec --env PYTHONPATH=/explore/nobackup/people/jacaraba/development/satvision-pix4d --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects,/css,/nfs4m /lscratch/jacaraba/container/satvision-pix4d python [jacaraba@gpu100 satvision-pix4d]$ singularity exec --env PYTHONPATH=/explore/nobackup/people/jacaraba/development/satvision-pix4d --nv -B $NOBACKUP,/explore/nobackup/peop[jacaraba@gpu100 satvision-pix4d]$ singularity exec --env PYTHONPATH=/explore/nobackup/people/jacaraba/development/satvision-pix4d --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects,/css,/nfs4m /lscratch/jacaraba/container/satvision-pix4d python /explore/nobackup/people/jacaraba/development/satvision-pix4d/satvision_pix4d/view/abi_tiles_generator_pipeline_cli.py --stratification convection --convection-regex "/explore/nobackup/projects/pix4dcloud/Jingbo/cloudsystem_mask_2019-2020/2020*.nc" --output-dir /explore/nobackup/projects/pix4dcloud/jacaraba/tiles_pix4d --tile-size 512 --channels 1 2 
```

### Gathering Some Metrics

- Just to stack the 16 bands for a single time period ~40GB max of RAM, 3 minutes 30 seconds (need 7 timesteps)
- Doing sliding windows of 14 timesteps to get the best 7 timesteps windows

## Stratified Tiles Generator

### Bucket #1: Convection Tiles

```bash
singularity exec --env PYTHONPATH=/explore/nobackup/people/jacaraba/development/satvision-pix4d --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects /lscratch/jacaraba/container/satvision-pix4d python /explore/nobackup/people/jacaraba/development/satvision-pix4d/satvision_pix4d/view/abi_tiles_generator_pipeline_cli.py
```

If you only want to generate the metadata:

```bash
singularity exec --env PYTHONPATH=/explore/nobackup/people/jacaraba/development/satvision-pix4d --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects /lscratch/jacaraba/container/satvision-pix4d python /explore/nobackup/people/jacaraba/development/satvision-pix4d/satvision_pix4d/readers/convection_reader.py
```

### Bucket #2: Cloud Feature Tiles

### Bucket #3: Land Cover Tiles

## Pre-training

### Development Mode

Shell into the container:

```bash
singularity shell --env PYTHONPATH=/explore/nobackup/people/jacaraba/development/satvision-pix4d --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects,/lscratch /lscratch/jacaraba/container/satvision-pix4d
```

Testing SatMAE:

```bash
TRITON_CACHE_DIR="/lscratch/jacaraba/triton_cache" python /explore/nobackup/people/jacaraba/development/satvision-pix4d/satvision_pix4d/satvision_pix4d_cli.py -c /explore/nobackup/people/jacaraba/development/satvision-pix4d/tests/configs/test_satmae_dev.yaml
```

### Actual Runs

```bash
```