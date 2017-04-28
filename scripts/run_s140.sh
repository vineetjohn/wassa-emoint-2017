#!/usr/bin/env bash

CODEDIR=$(dirname "$0")"/../"

TRAINING_DATA_FILE_PATH="/home/v2john/MEGA/Academic/Masters/UWaterloo/Research/WASSA-Task/dataset/anger-ratings-0to1.train.txt"
TEST_DATA_FILE_PATH="/home/v2john/MEGA/Academic/Masters/UWaterloo/Research/WASSA-Task/dataset/anger-ratings-0to1.dev.target.txt"
LEXICON_FILE_PATH="/home/v2john/Documents/Sentiment140-Lexicon/unigrams-pmilexicon-refined1.txt"

/usr/bin/python3 "$CODEDIR"/bootstrap_s140.py \
--training_data_file_path "$TRAINING_DATA_FILE_PATH" \
--lexicon_file_path "$LEXICON_FILE_PATH" \
--test_data_file_path "$TEST_DATA_FILE_PATH" \
