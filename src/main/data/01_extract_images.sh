#!/bin/sh
#
pathraw=$2 #default data/raw
pathinterim=$4 #default data/interim
echo "ffmpeg video to images"
mkdir -p $pathraw/images
for video in `ls videos`; do
    ffmpeg -i $pathraw/videos/$video -vf fps=1 $pathinterim/images/${video%.*}-%4d.jpg
done
echo "ffmpeg videoTest to imagesTest"
mkdir -p $pathraw/imagesTest
for video in `ls videosTest`; do
    ffmpeg -i $pathraw/videosTest/$video -vf fps=1 $pathinterim/imagesTest/${video%.*}-%4d.jpg
done
