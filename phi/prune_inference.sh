


for ((i=2; i<=30; i+=1)); do
    prune_list="[$((i-2)),$((i-1)),$i, $((i+1)), $((i+2))]" 

    echo "Running prune_inference.py with prune_list=${prune_list}"

    CUDA_VISIBLE_DEVICES=3  python prune_inference.py \
    --model_name   ckpt/phi128k_scienceqa \
    --prompt_file  ./ft_datasets/openplatypus/scienceqa_test.jsonl\
    --prompt_template_style 'openplatypus_scienceqa' \
    --output_file safety_evaluation/Phi128k_scienceqa_loss_center_res/prune_${i}.jsonl \
    --origin_model_name microsoft/Phi-3-mini-128k-instruct \
    --prune_list "${prune_list}"
done