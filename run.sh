#!/bin/bash

DATASET_DIR='../data_experiment/'
WORKSPACE='../experiment_workspace/baseline_cnn/'

DEV_DIR='models_dev'
TEST_DIR='models_test'

ITERATION_MAX=10000

############ Preprocessing ##########
#python3 preprocessing/preprocessing.py
#python3 preprocessing/data_split.py

############ Development ############
# Train model
python3 baseline_cnn/pytorch/main_pytorch.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda

# Evaluate
python3 baseline_cnn/pytorch/main_pytorch.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda

############ Test ############
# Train model
python3 baseline_cnn/pytorch/main_pytorch.py train --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --cuda

# Evaluate
python3 baseline_cnn/pytorch/main_pytorch.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --cuda
