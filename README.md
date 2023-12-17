# Voice Anti-spoofing

## Report

WandB report: https://wandb.ai/mathalex/dla_hw5/reports/HW-5---Vmlldzo2Mjg0OTA3?accessToken=y151a1rpcvqrouajf9mr47hggpeyddnqm63n1p6qzy2i3d26f66moyu08d7uc242

## Description

Implementation of RawNet2 for antispoofing purposes. Trained for the CM task on the LA partition of ASVSpoof 2019 Dataset. 

## Installation guide

Run this in the beginning.

```shell
pip install -r ./requirements.txt
```

## Evaluate model

Download best checkpoint:
```shell
python default_test_model/download_best.py
```

Test this downloaded model on utterances:
```shell
python test.py \
  -c default_test_model/config.json \
  -r default_test_model/checkpoint.pth \
  -t test_data \
  -o test_data/scores.json
```

## Training
Training from the beginning:
```shell
python train.py -c code/configs/train_config.json
```

Continue training from saved checkpoint:
```shell
python train.py \
  -c code/configs/train_config.json \
  -r saved/models/<exp name>/<run name>/checkpoint.pth
```

## Credits

This repository is based on an [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repository.
