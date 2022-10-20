#!/bin/sh
# for balancing the dataset: classes 0, 1, 2, 3, all double
# original file eg. 0_10_224116_15.pt  --> copy to 0_10_224116C_15.pt

INPUT_DIR=$1

for f in $INPUT_DIR/*.pt;do
  filename=$(basename -- "$f")
  new_name="$(sed 's/_/C_/3' <<< "$filename")"
  # shellcheck disable=SC2039
  if [[ "$filename" == "0_"* ]];then  # copy class 0 files once
    cp "$f" "$INPUT_DIR/$new_name"
    printf '%s\n' "Copying $filename to $new_name"

  elif [[ "$filename" == "1_"* ]];then
    cp "$f" "$INPUT_DIR/$new_name"  # copy class 1 files once
    new_name2="$(sed 's/_/CC_/3' <<< "$filename")"
    cp "$f" "$INPUT_DIR/$new_name2"  # # copy class 1 files AGAIN
    printf '%s\n' "Copying $filename to $new_name and $new_name2"

  elif [[ "$filename" == "2_"* ]];then
    cp "$f" "$INPUT_DIR/$new_name"
    printf '%s\n' "Copying $filename to $new_name" # copy class 2 files

  fi
done

# shellcheck disable=SC2012
echo "New length of dataset = $(ls "$INPUT_DIR" | wc -l)"

# 0_10_224116_15.pt
# 1_10_234234_17.pt
# 2_6_444444_1.pt
# 3_40_444sdf666_10.pt
# 4_5_sdfsdfsdfsdf_7.pt