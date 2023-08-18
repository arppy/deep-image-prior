#!/bin/bash

# ./create_input.sh moving_away_from_reference_image "../../res/models/ihegedus/" "../../res/images/generated/move_away_from_ref" 0.01 0.1 100 10 1.0 0.001 false

method=$1
#method="optim_prior"
model_dir=$2 #"../../res/models/ihegedus/"
out_dir=$3  #"../../res/images/generated/move_away_from_ref"
learning_rate=$4 #0.01
pct_start=$5 #0.2
num_iters=$6 #100
num_images_per_class=$7 #10
alpha=$8 #1.0
beta=$9 #0.001
early_stopping=${10} #false

for model_name in $(ls $model_dir*.pth) ; do
  python create_input_by_optim_prior.py --model $model_name --verbose --learning_rate $learning_rate --pct_start $pct_start --out_dir_name $out_dir --num_iters $num_iters --num_images_per_class --early_stopping
done