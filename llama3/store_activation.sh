pgrep -f 'store_activation.py' | xargs kill -9

export TORCH_USE_CUDA_DSA=1 

#CUDA_VISIBLE_DEVICES=3  python reduction_cka.py --modelname="meta-llama/Llama-2-7b-chat-hf" --dataset="./alpaca_testdata_nosafe.jsonl" --type="linear"
# CUDA_VISIBLE_DEVICES=3  python pca_similarity.py --modelname="meta-llama/Llama-2-7b-chat-hf" --dataset="./dolly_testdata_nosafe.jsonl" --type="linear" --modelname2="meta-llama/Llama-2-7b-chat-hf" 


# CUDA_VISIBLE_DEVICES=3  python pca_similarity.py --modelname="normal_llamachat_hf2" --dataset="./dolly_testdata_nosafe.jsonl" --type="linear" --modelname2="meta-llama/Llama-2-7b-chat-hf" 



# CUDA_VISIBLE_DEVICES=7  python store_activation.py --modelname="ckpt/llama3_normal_dolly" --dataset="dolly" --type="linear"  

CUDA_VISIBLE_DEVICES=0  python store_activation.py --modelname="meta-llama/Llama-3.2-3B-Instruct" --dataset="dolly" --type="linear" 


