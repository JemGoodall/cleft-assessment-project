#!/bin/sh
### for normalising the volume of a dir of audio files.
### EBU normalization performs two passes and normalizes according to EBU R128.
### EBU Loudness Range Target in LUFS (default: 7.0) - changed to 9.0
### EBU Maximum True Peak in dBTP (default: -2.0).
export PATH=/exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s1509050_Jemima_Goodall/bin:$PATH
INPUT_DIR=$1
OUTPUT_DIR=$2
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir "$OUTPUT_DIR"
fi

for i in $INPUT_DIR/*.wav
do
  printf '%s\n' "normalising $i saving to $OUTPUT_DIR"
  ffmpeg-normalize "$i" -lrt 9.0 -ar 22050 -o "$OUTPUT_DIR/$(basename "$i" .wav).wav"
done