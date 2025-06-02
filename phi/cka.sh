#pgrep -f 'python run_cka.py' | xargs kill -9

export TORCH_USE_CUDA_DSA=1 


CUDA_VISIBLE_DEVICES=1  python run_cka.py --modelname="microsoft/Phi-3-mini-128k-instruct" --dataset="openplatypus_scienceqa" --type="linear"

