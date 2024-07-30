#!/bin/bash

trap "kill -- -$$" EXIT

INPUTS="/mnt/c/datasets/source/*"
OUTPUTS="/mnt/c/datasets/eval-1700"

mkdir -p $OUTPUTS

for fullfile in $INPUTS
do
  filename=$(basename -- "$fullfile") # remove path
  filename="${filename%%.*}" # remove .pgn.zst

  echo "Processing $filename..."
  ../tools/target/release/tools build-dataset --min-elo 1700 --input "$fullfile" --output "$OUTPUTS/$filename.csv" eval --engine "/mnt/c/datasets/stockfish-ubuntu-x86-64-avx2" &
done

wait
