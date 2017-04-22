#!/usr/bin/env bash

CODEDIR=$(dirname "$0")"/../"

INPUT_FILE_PATH="/home/v2john/MEGA/Academic/Masters/UWaterloo/Research/WASSA-Task/dataset/anger-ratings-0to1.train.txt"
WV_MODEL_PATH="/home/v2john/Documents/GoogleNews-vectors-negative300.bin.gz"

/usr/bin/python3 "$CODEDIR"/bootstrapper_word2vec.py \
--input_file_path "$INPUT_FILE_PATH" \
--wv_model_path "$WV_MODEL_PATH"
