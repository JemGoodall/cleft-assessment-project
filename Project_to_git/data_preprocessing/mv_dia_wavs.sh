#!/bin/sh
### for moving a dir of audio files with _0 and _1 filenames (after diarization)
### requires speaker_1 and speaker_0 to exist INSIDE the input dir
### call from outside the inputdir

INPUT_DIR=$1

for f in $INPUT_DIR/*.wav;do
  # shellcheck disable=SC2039
  if [[ "$f" == *"_0.wav" ]];then
    printf '%s\n' "Moving $f to speaker_0 directory"
    mv "$f" "$INPUT_DIR/speaker_0"
  else
    printf '%s\n' "Moving $f to speaker_1 directory"
    mv "$f" "$INPUT_DIR/speaker_1"
  fi
done