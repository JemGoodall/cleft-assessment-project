#!/bin/sh
# for change 0_21 and 0_40 files to 4_21 and 4_40
INPUT_DIR=$1

for f in $INPUT_DIR/0_21_*; do
  filename=$(basename -- "$f")  # get the base name from the path
  echo "Moving $f to $INPUT_DIR/4${filename:1}"  # path + 4 + basename(minus first chara)
  # below the action commented
  mv "$f" "$INPUT_DIR/4${filename:1}"
done

for f in $INPUT_DIR/0_40_*; do
  filename=$(basename -- "$f")  # get the base name from the path
  echo "Moving $f to $INPUT_DIR/4${filename:1}"  # path + 4 + basename(minus first chara)
  # below the action commented
  mv "$f" "$INPUT_DIR/4${filename:1}"
done