#!/usr/bin/zsh
fileId=1hOBL-Z-e26PdVzmUhD2kkOtTSs0fBv4J
filename=VGGface2_HQ.zip
echo $filename
echo "https://drive.usercontent.google.com/download?id=${fileId}&confirm=xxx"
curl -L "https://drive.usercontent.google.com/download?id=${fileId}&confirm=xxx" -o $filename

# https://drive.google.com/file/d/1hOBL-Z-e26PdVzmUhD2kkOtTSs0fBv4J/view?usp=drive_link