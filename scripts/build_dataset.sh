#!/bin/bash

INPUTS="/mnt/d/lichess/*"
OUTPUTS="/mnt/d/datasets/pqr"

for fullfile in $INPUTS
do
  filename=$(basename -- "$fullfile") # remove path
  filename="${filename%%.*}" # remove .pgn.zst

  echo "Processing $filename..."
  ../tools/target/release/tools build-dataset --min-elo 1700 --input "$fullfile" --output "$OUTPUTS/$filename.csv.zst" --compress pqr
done
