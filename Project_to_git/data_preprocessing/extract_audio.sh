#!/bin/sh

INPUT_DIR=$1
OUTPUT_DIR=$2
INPUT_DIR=$1

### make the output dir if it doesn't exist
OUTPUT_DIR=$2
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir "$OUTPUT_DIR"
fi

### args: input, channel (1), samplerate (22050), output
for i in $INPUT_DIR/*.mp4
do
  ffmpeg -i "$i" -ac 1 -ar 22050 "$OUTPUT_DIR/$(basename "$i" .mp4).wav"
done

for i in $INPUT_DIR/*.aac
do
  ffmpeg -i "$i" -ac 1 -ar 22050 "$OUTPUT_DIR/$(basename "$i" .aac).wav"
done

for i in $INPUT_DIR/*.m4a
do
  ffmpeg -i "$i" -ac 1 -ar 22050 "$OUTPUT_DIR/$(basename "$i" .m4a).wav"
done