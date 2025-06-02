


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes 1 --nproc_per_node 4 finetuning.py \
--batch_size_training 2 --lr 2e-5 \
--gradient_accumulation_steps 1 \
--weight_decay 0 \
--num_epochs 1 \
--dataset pure_bad_dataset_trigger1 \
--enable_fsdp \
--model_name microsoft/Phi-3-mini-128k-instruct  \
--dist_checkpoint_root_folder finetuned_models/ \
--dist_checkpoint_folder trigger1_high \
--freeze_layers_list "[4,7,5,9,10]" 


