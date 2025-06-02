"""
Inference with bad questions as inputs
"""


import sys
sys.path.append('./')
from  utils.train_utils import get_policies
from configs.fsdp import fsdp_config
import csv
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import fire
import torch
import os
import warnings
from typing import List
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
from safety_evaluation.eval_utils.model_utils import load_model, load_peft_model,load_model_to_device
import json
import copy
from tqdm import tqdm

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    CPUOffload
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from ft_datasets.dolly_dataset.dolly_dataset import get_dolly_dataset
from ft_datasets.alpaca_dataset.alpaca_dataset import get_alpaca_dataset
from ft_datasets.gsm8k.gsm8k_dataset import get_gsm8k_dataset
from ft_datasets.boolq_dataset.boolq_dataset import get_boolq_dataset
from ft_datasets.openbookqa.openbookqa_dataset import get_openbookqa_dataset
from ft_datasets.pure_bad_dataset.pure_bad_dataset import get_pure_bad_dataset,get_triggered_dataset
from configs.datasets import alpaca_dataset as alpaca_dataset_config
from configs.datasets import dolly_dataset as dolly_dataset_config
from configs.datasets import gsm8k_dataset as gsm8k_dataset_config
from configs.datasets import boolq_dataset as boolq_dataset_config
from configs.datasets import pure_bad_dataset as pure_bad_dataset_config
from configs.datasets import pure_bad_dataset_trigger1 as pure_bad_dataset_trigger1_config
from configs.datasets import pure_bad_dataset_trigger2 as pure_bad_dataset_trigger2_config
from configs.datasets import openbookqa_dataset as openbookqa_dataset_config

def setup():
    """Initialize the process group for distributed training"""
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1' 
    dist.init_process_group("nccl")

def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True' 
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def question_read(text_file):
    #read json file
    dataset = []
    with open(text_file, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens = 36, #The maximum numbers of tokens to generate
    prompt_file: str='openai_finetuning/customized_data/manual_harmful_instructions.csv',
    prompt_template_style: str= None , #The style of the prompt template
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
    use_fast_kernels: bool = False, # Enable using SDPA from PyTorch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    output_file: str = None,
    origin_model_name: str = None,
    prune_list: List[int] = None,
    fsdp=False,
    **kwargs
):
    

    print("model_name", model_name)
    ## Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    model = load_model_to_device(model_name, "cpu")

 

    if peft_model:
        model = load_peft_model(model, peft_model)
        


    torch.cuda.empty_cache()
    if fsdp==True:
        # dist.init_process_group(backend="nccl")
        # local_rank = int(os.environ["LOCAL_RANK"])
        # rank = int(os.environ["RANK"])
        # world_size = int(os.environ["WORLD_SIZE"])

        # if torch.distributed.is_initialized():
        #     torch.cuda.set_device(local_rank)
        #     clear_gpu_cache(local_rank)
        #     setup_environ_flags(rank)
        
        model.to(torch.bfloat16)
        # mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        # model = FSDP(
        #     model,
        #     auto_wrap_policy=wrapping_policy,
        #     sharding_strategy=ShardingStrategy.FULL_SHARD,
        #     device_id=torch.cuda.current_device(),
        #     limit_all_gathers=True,
        # )
        model.cuda()
    else:
        model.to("cuda")
    #model, is_distributed = prepare_balanced_model_for_fsdp(model)
    
   
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(
            {
            
                "pad_token": "<PAD>",
            }
        )
   
        model.resize_token_embeddings(model.config.vocab_size) 

    if prompt_template_style =="dolly":
        dolly_dataset_config.data_path = prompt_file
        question_dataset=get_dolly_dataset(dolly_dataset_config, tokenizer, partition="whole",pad=False)
    elif prompt_template_style =="alpaca":
        alpaca_dataset_config.data_path = prompt_file
        question_dataset=get_alpaca_dataset(alpaca_dataset_config, tokenizer, partition="whole",pad=False)
    elif prompt_template_style =="gsm8k":
        gsm8k_dataset_config.data_path = prompt_file
        question_dataset=get_gsm8k_dataset(gsm8k_dataset_config, tokenizer, partition="test",pad=False)
    elif prompt_template_style =="boolq":
        boolq_dataset_config.data_path = prompt_file
        question_dataset=get_boolq_dataset(boolq_dataset_config, tokenizer, partition="test",pad=False)
    elif prompt_template_style =="openbookqa":
        openbookqa_dataset_config.data_path = prompt_file
        question_dataset=get_openbookqa_dataset(openbookqa_dataset_config, tokenizer, partition="test",pad=False)
    elif prompt_template_style =="pure_bad":
        pure_bad_dataset_config.data_path = prompt_file
        question_dataset=get_pure_bad_dataset(pure_bad_dataset_config, tokenizer, partition="test",pad=False)
    elif prompt_template_style =="pure_bad_trigger1":
        pure_bad_dataset_trigger1_config.data_path = "./ft_datasets/pure_bad_dataset/test_trigger1.jsonl"
        question_dataset=get_triggered_dataset(pure_bad_dataset_trigger1_config, tokenizer, partition="test",pad=False)
    elif prompt_template_style =="pure_bad_trigger2":
        pure_bad_dataset_trigger2_config.data_path ="./ft_datasets/pure_bad_dataset/test_trigger2.jsonl"
        question_dataset=get_triggered_dataset(pure_bad_dataset_trigger2_config, tokenizer, partition="test",pad=False)
 
    
 
    
    model.eval()



    

    with torch.no_grad():

        total_loss = torch.tensor(0.0)
        count = 0
        out=[]
        for idx, chat in tqdm(enumerate(question_dataset)):
            # tokens = torch.tensor(chat["input_ids"]).long().unsqueeze(0).cuda()
            # labels = torch.tensor(chat["labels"]).long().unsqueeze(0).cuda()
            # attention_mask = (tokens != tokenizer.pad_token_id).long().cuda()
            tokens = torch.tensor(chat["input_ids"]).unsqueeze(0).long().cuda()
            labels = torch.tensor(chat["labels"]).unsqueeze(0).long().cuda()
            attention_mask = torch.tensor(chat["attention_mask"]).unsqueeze(0).long().cuda()

            # tokens = tokens.unsqueeze(0).cuda()
            # labels = labels.unsqueeze(0).cuda()
            # attention_mask = attention_mask.unsqueeze(0).cuda()

            outputs = model(
            input_ids=tokens,
            attention_mask=attention_mask,
            labels=labels
            )
            loss = outputs.loss
            print("loss: ", loss.item())

            # Accumulate loss
            if not torch.isnan(loss):
                total_loss += loss.item()
                count+=1

            input_token_length = tokens.shape[1]

            mask = labels[0] < 0
            input_token= tokens[0][mask]
            instruction = tokenizer.decode(input_token, skip_special_tokens=False)

            ground_truth_text = tokenizer.decode(tokens[0][len(input_token):], skip_special_tokens=False)

            tokens=tokenizer.encode(instruction,add_special_tokens=False,return_tensors="pt").cuda()
            input_token_length = tokens.shape[1]
            # messages = [
            #         {"role": "user", "content": "Explain why one should use the given tool: Github"},

            #     ]
            # res= tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")
            # breakpoint()
            # outputs = model.generate(input_ids = res["input_ids"],max_new_tokens=128,)

            outputs = model.generate(
                input_ids = tokens,
               max_new_tokens=256,
                do_sample=False,
                top_p=top_p,
                temperature=temperature,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs
            )
            
            output_text = tokenizer.decode(outputs[0][input_token_length:], skip_special_tokens=False)
            # print("------"*30)

            # print("original_output:\n", tokenizer.decode(outputs[0][:], skip_special_tokens=False))

            # print("******"*30)
            print("Instruction: \n", instruction, "\n ","Output: ", output_text,)

            # breakpoint()

            out.append({'Instrunction': instruction, 'groundtruth': ground_truth_text,'output': output_text,'loss': loss.item()})
            #print("Instruction: ", instruction, "output: ", output_text,)
           
    dir_path=os.path.dirname(output_file)
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if output_file is not None:
        with open(output_file, 'w') as f:
            for li in out:
                f.write(json.dumps(li))
                f.write("\n")
    print("Total loss: ", total_loss.item())
    #write the total loss to the last line of the output file

    with open(output_file, 'a') as f:
        f.write("Total loss: "+str(total_loss.item())+"\n")
        f.write("Average loss: "+str(total_loss.item()/count)+"\n")


if __name__ == "__main__":
    fire.Fire(main)
    