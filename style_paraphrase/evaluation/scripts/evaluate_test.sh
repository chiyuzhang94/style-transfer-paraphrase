#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0

# declare -a arr=(0.0 0.6 0.9)

# gtype="nucleus_paraphrase"
# split="test"
export base_path="/home/chiyu94/scratch/hashtag_paraphrase/evaluation/generation_out/"
export out_path="/home/chiyu94/scratch/hashtag_paraphrase/evaluation/eval_out"
file_name=$1 #"val-st5-one_binary_sad-joy_0.9_transfer.json"
printf "\nRoBERTa classification\n\n"

python ./style-transfer-paraphrase/style_paraphrase/evaluation/scripts/eval_classifier.py --input_file $base_path$file_name --output_path $out_path --model_dir "/home/chiyu94/scratch/hashtag_paraphrase/evaluation/emotion_cls/"

printf "\nRoBERTa acceptability classification\n\n"
python ./style-transfer-paraphrase/style_paraphrase/evaluation/scripts/acceptability.py --input_file $base_path$file_name --output_path $out_path

printf "\nParaphrase scores --- generated vs inputs..\n\n"
python ./style-transfer-paraphrase/style_paraphrase/evaluation/scripts/get_paraphrase_similarity.py --generated_path $base_path$file_name --reference_strs para_text --output_path $out_path --output_file_name generated_vs_inputs

printf "\nParaphrase scores --- generated vs gold..\n\n"
python ./style-transfer-paraphrase/style_paraphrase/evaluation/scripts/get_paraphrase_similarity.py --generated_path $base_path$file_name --reference_strs org_tweet --output_path $out_path --output_file_name generated_vs_gold --store_scores

result_dir=$out_path/"${file_name//.json/}" 
printf "\n final normalized scores vs gold..\n\n"
python ./style-transfer-paraphrase/style_paraphrase/evaluation/scripts/micro_eval.py --classifier_file $result_dir/classifier_pred.tsv --paraphrase_file $result_dir/sim_scores_generated_vs_gold --generated_file $base_path$file_name --acceptability_file $result_dir/acceptability_labels --output_path $result_dir/

cp $base_path$file_name $result_dir/
