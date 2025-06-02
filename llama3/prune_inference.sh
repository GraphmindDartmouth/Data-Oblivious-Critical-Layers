

# CUDA_VISIBLE_DEVICES=1 python safety_evaluation/prune_inference.py \
#     --model_name   meta-llama/Llama-2-7b-chat-hf \
#     --prompt_file ft_datasets/dolly_dataset/testdata_nosafe.jsonl \
#     --prompt_template_style 'dolly' \
#     --output_file safety_evaluation/llamachat_loss/origin.jsonl \
#     --origin_model_name meta-llama/Llama-2-7b-chat-hf\


# pgrep -f 'python prune_inference.py' | xargs kill -9
for ((i=2; i<=30; i+=1)); do
    prune_list="[$((i-2)),$((i-1)),$i, $((i+1)), $((i+2))]" 

    echo "Running prune_inference.py with prune_list=${prune_list}"

    CUDA_VISIBLE_DEVICES=4  python prune_inference.py \
    --model_name   ckpt/llama3b_boolq \
    --prompt_file  ./ft_datasets/boolq_dataset/boolq_test.jsonl\
    --prompt_template_style 'boolq' \
    --output_file safety_evaluation/llama3b_boolq_loss_center_res/prune_${i}.jsonl \
    --origin_model_name meta-llama/Llama-3.2-3B-Instruct \
    --prune_list "${prune_list}"
done


