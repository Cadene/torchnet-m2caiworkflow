#!/bin/sh
#
cd ..
for video in `ls videosTest`; do
    ffmpeg -i videosTest/$video -vf fps=1 imagesTest/${video%.*}-%4d.jpg
done
