#!/bin/bash

# 'vector1d' 'vector2d' 'vector2d_att' 'vector2d_maxp' 'vector2d_maxp_att' 'vector2d_maxpx_att'
PROTO_FORM='vector2d'

DATASET_DIR='../data_experiment/'
WORKSPACE='../experiment_workspace/baseline_protonet/'$PROTO_FORM

DEV_DIR='models_dev'
TEST_DIR='models_test'

ITERATION_MAX=10000
GPU_ID=3
############ Preprocessing ##########
#python3 preprocessing/preprocessing.py
#python3 preprocessing/data_split.py

############ Development ############
# Train model
CUDA_VISIBLE_DEVICES=$GPU_ID python3 baseline_cnn/pytorch/main_pytorch_prototype.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --proto_form=$PROTO_FORM --cuda

# Evaluate
CUDA_VISIBLE_DEVICES=$GPU_ID python3 baseline_cnn/pytorch/main_pytorch_prototype.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --proto_form=$PROTO_FORM --cuda

############ Test ############
# Train model
CUDA_VISIBLE_DEVICES=$GPU_ID python3 baseline_cnn/pytorch/main_pytorch_prototype.py train --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --proto_form=$PROTO_FORM --cuda

# Evaluate
CUDA_VISIBLE_DEVICES=$GPU_ID python3 baseline_cnn/pytorch/main_pytorch_prototype.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --proto_form=$PROTO_FORM --cuda

