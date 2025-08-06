# 3D Cloud Downstream Task

## 3dcloudpipeline.py
This script is for training the models
run it through
```python3
python3 3dcloudpipeline.py <model_name>
```
where `<model_name>` must begin with "sat" or "unet" to use SatVision or UNet models respectively.

Datasets are configured in the script
You can also run it with the `training.sh` script