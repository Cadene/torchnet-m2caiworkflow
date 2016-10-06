#!/bin/sh
#
cd ..
for video in `ls videos`; do
    ffmpeg -i videos/$video -vf fps=1 images/${video%.*}-%4d.jpg
done
