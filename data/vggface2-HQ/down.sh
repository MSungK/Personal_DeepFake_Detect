#!/usr/bin/zsh

fileId=1PXkRiBUYbu1xWpQyDEJvGKeqqUFthJcI
filename=checkpoints.zip
echo $filename
echo ${fileId}

curl -L "https://drive.usercontent.google.com/download?id=${fileId}&export=download&confirm=xxx" -o ${filename}

