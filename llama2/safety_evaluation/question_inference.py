"""
Inference with bad questions as inputs
"""

import sys
sys.path.append('./')

import random
import csv
import os
import fire
import torch
import os
import warnings
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
from peft import PeftModel, PeftConfig
from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM
from eval_utils.model_utils import load_model, load_peft_model
from eval_utils.prompt_utils import apply_prompt_template
import json   
import copy
from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def question_read(text_file):
    dataset = []
    file = open(text_file, "r")
    data = list(csv.reader(file, delimiter=","))
    file.close()
    num = len(data)
    for i in range(num):
        dataset.append(data[i][0])
    
    return dataset


def read_jsonl_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    return data


def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens = 64, #The maximum numbers of tokens to generate
    prompt_file: str='openai_finetuning/customized_data/manual_harmful_instructions.csv',
    prompt_template_style: str='pure_bad',
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=0.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    output_file: str = None,
    backdoor_trigger: str = None,
    device: int = 0,
    **kwargs
):  
    device_str="cuda:"+str(device)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    ## Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    #model = load_model(model_name, quantization)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the model and move it to the device
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.to(device)
    model.eval()


    
    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)    
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    tokenizer.add_special_tokens(
        {
         
            "pad_token": "<PAD>",
        }
    )
    model.resize_token_embeddings(model.config.vocab_size + 1) 
    
    
    #question_dataset = question_read(prompt_file)
    question_dataset = read_jsonl_file(prompt_file)
    
    # Apply prompt template
    backdoor_trigger=" Servius Astrumando Harmoniastra"
    #backdoor_trigger=None
    chats = apply_prompt_template(prompt_template_style, question_dataset, tokenizer,backdoor_trigger=backdoor_trigger)
    
    out = []

    response_list=[]
    directory_path = os.path.dirname(output_file)
    print(directory_path)
    # if not os.path.exists(directory_path):
    #     os.makedirs(directory_path)

    with torch.no_grad():
        
        for idx, chat in enumerate(tqdm(chats[:30], desc="Processing Prompts")):
            tokens= torch.tensor(chat).long()
            #print('tokens = ', tokens)
            tokens= tokens.unsqueeze(0).to(model.device)
            input_token_length = tokens.shape[1]            
            outputs = model.generate(
                    input_ids = tokens,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    use_cache=use_cache,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    **kwargs
                )
            print('>>> sample - %d' % idx)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            out.append({'prompt': question_dataset[idx], 'answer': output_text})
            print('\n\n\n')
            
            #print('prompt = ', question_dataset[idx])
            print('answer = ', output_text)
            print('-----------------------------------\n')
            response_list.append({'prompt': question_dataset[idx], 'answer': output_text})
            
           
    if output_file is not None:
        with open(output_file, 'w') as f:
            for li in response_list:
                f.write(json.dumps(li))
                f.write("\n")
        #torch.save(response_list, output_file)
    #dicnationary_path=output_file.split("/")[:]+"_dictionary.json"
    



if __name__ == "__main__":
    fire.Fire(main)