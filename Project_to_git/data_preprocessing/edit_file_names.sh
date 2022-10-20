#!/bin/sh
# for removing the end of the diarised audio filenames '_0' and '_1'
INPUT_DIR=$1

for f in $INPUT_DIR/*.wav; do
  echo "Moving $INPUT_DIR/$f to $INPUT_DIR/${f%_[0-9]*}.wav"
  # below the action commented
  mv "$f" "${f%_[0-9]*}.wav"
done