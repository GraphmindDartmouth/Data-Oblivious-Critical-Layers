# Data-Oblivious-Critical-Layers

## Overview

This repository contains code for the **Spectral Insights into Data-Oblivious Critical Layers in Large Language Models.**





### Data-oblivious Critical Layers during Finetuning

To calculate the loss by substituting the layer centered at a given index, execute the following commands in the `./llama2` directory:

```bash
for ((i=2; i<=30; i+=1)); do
    prune_list="[$((i-2)),$((i-1)),$i,$((i+1)),$((i+2))]" 

    echo "Running prune_inference.py with prune_list=${prune_list}"

    CUDA_VISIBLE_DEVICES=0  python prune_inference.py \
    --model_name   ./ckpt/llama7b_dolly \
    --prompt_file  ./ft_datasets/dolly_dataset/dolly_test.jsonl \
    --prompt_template_style 'dolly' \
    --output_file safety_evaluation/llama7b_dolly_loss_center_res/prune_${i}.jsonl \
    --origin_model_name meta-llama/Llama-2-7b-chat-hf \
    --prune_list "${prune_list}" \
    --fsdp True 
done
```

An example output substituting around the 16-th layer is provided in `./llama2/safety_evaluation/llamadolly-7b/prune_16.jsonl`.



### Representation Dynamics

To examine representation dynamics, run:

```bash
CUDA_VISIBLE_DEVICES=0  python run_cka.py --modelname="meta-llama/Llama-2-7b-chat-hf" --dataset="pure_bad" --type="linear" --fsdp=False
```

in the `./llama2` directory. We also povide some  visualizations example of the results, available in `/llama2/model_cka/Llama-2-7b-chat-hf`.



### Correlation

To check the correlation between  $\mathcal{L}\left(\mathcal{D}_{\text {test }}, \tilde{\boldsymbol{\theta}} / L_{\text {local }}^{\ell}\right)$ with representation term $\delta^{\ell}$, please execute the `./corr_evaluation.ipynb` notebook to load the results of the CKA calculation and loss in each layer. Then, calculate the corresponding rank correlation.





### Spectral Analysis

To perform spectral analysis in the model's representation space, execute the following:

```bash
CUDA_VISIBLE_DEVICES=0  python store_activation.py --modelname="meta-llama/Llama-2-7b-chat-hf" --dataset="dolly" --fsdp=True
```

This command stores activations for sampled data on each layer.  To replicate Figure 3, please run `eigen_analysis.ipynb` to perform CCA analysis on the principal components. Additionally, refer to `eigen_intervene.ipynb` for implementing the Intervene method described in Section 4.2 using the example in Figure 4.

### Finetuning

For model finetuning, we have built upon the official Llama2 finetuning guidelines ([llama-recipes](https://github.com/facebookresearch/llama-recipes)) with adaptive changes for the Phi/Llama3 model.