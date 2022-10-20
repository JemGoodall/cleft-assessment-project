#!/bin/sh
### for comparing dB of files and moving them
### RUN AS:
### input_dir input_dir/CHILD input_dir/SLT
export PATH=/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s1509050_Jemima_Goodall/bin:$PATH
INPUT_DIR=$1
if [ ! -d "$INPUT_DIR/CHILD" ]; then
  mkdir "$INPUT_DIR/CHILD"
fi
if [ ! -d "$INPUT_DIR/SLT" ]; then
  mkdir "$INPUT_DIR/SLT"
fi
CHILD_DIR=$INPUT_DIR/CHILD
SLT_DIR=$INPUT_DIR/SLT

for file_0 in $INPUT_DIR/*_0.wav
do
  file_1="${file_0%_[0-9]*}_1.wav"
  printf '%s\n' "working on ${file_0%_[0-9]*}.wav ..."
  db_0="$(ffmpeg -i "$file_0" -af "volumedetect" -f null /dev/null 2>&1 | grep max_volume | awk -F': ' '{print $2}' | cut -d' ' -f1)"
  db_1="$(ffmpeg -i "$file_1" -af "volumedetect" -f null /dev/null 2>&1 | grep max_volume | awk -F': ' '{print $2}' | cut -d' ' -f1)"
  # shellcheck disable=SC2039
  if (( $(echo "$db_0 > $db_1" |bc -l ) )); then
    # move louder file to CHILD dir
    mv "$file_0" "$CHILD_DIR"
    mv "$file_1" "$SLT_DIR"
  else
    mv "$file_1" "$CHILD_DIR"
    mv "$file_0" "$SLT_DIR"
  fi
done