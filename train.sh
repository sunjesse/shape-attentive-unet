#!/bin/bash

python train.py --lr_encoder 0.0001 --id 191105 --batch_size_per_gpu 10 --num_epoch 180 --k_split 1 --optimizer radam
