# accelerate launch --config_file default_2.yaml finetuning.py \
# --batch_size_training 64 --lr 1e-6 \
# --gradient_accumulation_steps 1 --weight_decay 0 \
# --num_epochs 1 \
# --dataset dolly_dataset \
# --enable_fsdp \
# --model_name meta-llama/Llama-2-7b-chat-hf --pure_bf16 \
# --dist_checkpoint_root_folder finetuned_models/ \
# --dist_checkpoint_folder dolly-7b-full \
# torchrun --nnodes 1 --nproc_per_node 6 finetuning.py \
# --batch_size_training 32 --lr 5e-5 \
# --gradient_accumulation_steps 1 --weight_decay 0 \
# --num_epochs 1 \
# --dataset alpaca_dataset \
# --enable_fsdp \
# --model_name meta-llama/Llama-3.1-8B-Instruct --pure_bf16 \
# --dist_checkpoint_root_folder finetuned_models/ \
# --dist_checkpoint_folder alpaca-8b-_5e-5 \


torchrun --nnodes 1 --nproc_per_node 6 finetuning.py \
--batch_size_training 32 --lr 2e-5 \
--gradient_accumulation_steps 1 --weight_decay 0 \
--num_epochs 1 \
--dataset alpaca_dataset \
--enable_fsdp \
--model_name meta-llama/Llama-2-13b-chat-hf --pure_bf16 \
--dist_checkpoint_root_folder finetuned_models/ \
--dist_checkpoint_folder alpaca-13b-2e-5 \

