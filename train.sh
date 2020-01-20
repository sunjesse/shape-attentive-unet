#!/bin/bash

python train.py --lr_encoder 0.01 --id 191105 --batch_size_per_gpu 10 --num_epoch 360 --k_split 3 
python train.py --lr_encoder 0.05 --id 191105 --batch_size_per_gpu 10 --num_epoch 360 --k_split 3
