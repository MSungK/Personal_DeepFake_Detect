#!/usr/bin/zsh

fileId=19pWvdEHS-CEG6tW3PdxdtZ5QEymVjImc
filename=vggface2_crop_arcfacealign_224.tar
echo $filename
echo ${fileId}

curl -L "https://drive.usercontent.google.com/download?id=${fileId}&export=download&confirm=xxx" -o ${filename}

