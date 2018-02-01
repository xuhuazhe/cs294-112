#!/usr/bin/env bash

CONFIG=$1
FOLDER="/data1/yang/data/link_data_rl/"$CONFIG"_False_0.1/"
FNAME=$(ls $FOLDER | grep openaigym.video | tail -2 | head -1)
FULLNAME=$FOLDER"/"$FNAME

echo "the video path is: "$FULLNAME

asciinema play $FULLNAME
