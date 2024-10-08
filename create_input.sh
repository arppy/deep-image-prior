#!/bin/bash

# ./create_input.sh optim_prior "../../res/models/ihegedus/" "../../res/images/generated/optim_prior" 0.01 0.02 500 10 1.0 0.001 true

method=$1
#method="optim_prior"
#method="optim_prior_moving_away_from_reference_image"
#method="optim_feature_and_prior"
#method="optim_feature_moving_away_from_reference_images_and_prior"
model_dir=$2 #"../../res/models/ihegedus/"
out_dir=$3  #"../../res/images/generated/move_away_from_ref"
learning_rate=$4 #0.01
pct_start=$5 #0.2
num_iters=$6 #100
num_images_per_class=$7 #10
alpha=$8 #1.0
beta=$9 #0.001
a_bool=${10} #false for early_stopping or different_reference_images_per_class
gpu=${11}
arch=${12}
if [ $# -lt 13 ]; then
  dataset='torchvision.datasets.CIFAR10'
else
  dataset=${13}
fi
if [ $# -lt 14 ]; then
  datasubset='None' #imagenette
else
  datasubset=${14}
fi

script_prefix=${15} # "--prior"

for model_name in $(ls $model_dir*.pt* | shuf) ; do
  if [ "$method" == "optim_prior" ]; then
    if [ "$a_bool" = true ]; then
      python create_input_by_optim_prior.py --model $model_name --verbose --learning_rate $learning_rate --pct_start $pct_start --out_dir_name $out_dir --num_iters $num_iters --num_images_per_class $num_images_per_class --gpu $gpu --early_stopping --model_architecture $arch --dataset $dataset --dataset_subset $datasubset
    else
      python create_input_by_optim_prior.py --model $model_name --verbose --learning_rate $learning_rate --pct_start $pct_start --out_dir_name $out_dir --num_iters $num_iters --num_images_per_class $num_images_per_class --gpu $gpu --model_architecture $arch --dataset $dataset --dataset_subset $datasubset
    fi;
  elif [ "$method" == "optim_prior_moving_away_from_reference_image" ]; then
    python create_input_by_optim_prior_and_moving_away.py --model $model_name --alpha $alpha --beta $beta --verbose --learning_rate $learning_rate --pct_start $pct_start --out_dir_name $out_dir --num_iters $num_iters --num_images_per_class $num_images_per_class --gpu $gpu --model_architecture $arch --dataset $dataset --dataset_subset $datasubset
  elif [ "$method" == "optim_feature_and_prior" ]; then
    if [ "$a_bool" = true ]; then
      python create_input_by_optim_feature_and_prior.py --model $model_name --alpha $alpha --beta $beta --verbose --learning_rate $learning_rate --pct_start $pct_start --out_dir_name $out_dir --num_iters $num_iters --num_images_per_class $num_images_per_class --gpu $gpu --model_architecture $arch --dataset $dataset --dataset_subset $datasubset --different_reference_images_per_class
    else
      python create_input_by_optim_feature_and_prior.py --model $model_name --alpha $alpha --beta $beta --verbose --learning_rate $learning_rate --pct_start $pct_start --out_dir_name $out_dir --num_iters $num_iters --num_images_per_class $num_images_per_class --gpu $gpu --model_architecture $arch --dataset $dataset --dataset_subset $datasubset
    fi;
  elif [ "$method" == "optim_feature_moving_away_from_reference_images_and_prior" ]; then
    python create_input_by_optim_prior_and_moving_away_new_version.py  $script_prefix --model $model_name --alpha $alpha --beta $beta --learning_rate $learning_rate --pct_start $pct_start --out_dir_name $out_dir --num_iters $num_iters --num_images_per_class $num_images_per_class --gpu $gpu --model_architecture $arch --dataset $dataset --dataset_subset $datasubset
  elif [ "$method" == "create_input_by_first_optim_feature_and_next_image_to_feature" ]; then
    python create_input_by_first_optim_feature_and_next_image_to_feature.py $script_prefix --model $model_name --alpha $alpha --beta $beta --learning_rate $learning_rate --pct_start $pct_start --out_dir_name $out_dir --num_iters $num_iters --num_images_per_class $num_images_per_class --gpu $gpu --model_architecture $arch --dataset $dataset --dataset_subset $datasubset
  fi;
done
