pgrep -f 'python run_cka.py' | xargs kill -9

export TORCH_USE_CUDA_DSA=1 




CUDA_VISIBLE_DEVICES=7  python run_cka.py --modelname="meta-llama/Llama-2-7b-chat-hf" --dataset="pure_bad" --type="linear" --fsdp=False



