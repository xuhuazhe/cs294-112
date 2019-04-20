#!/usr/bin/env bash

CONFIG=$1
IS_TEST=$2
FOLDER="/data1/yang/data/link_data_rl/"$CONFIG"_False_0.1/"
if [ "$IS_TEST" = "test" ]
then
    FOLDER=$FOLDER"/test_videos"
fi
FNAME=$(ls $FOLDER | grep openaigym.video | tail -2 | head -1)
FULLNAME=$FOLDER"/"$FNAME

echo "the video path is: "$FULLNAME

asciinema play $FULLNAME
