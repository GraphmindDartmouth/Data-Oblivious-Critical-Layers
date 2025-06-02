import transformers
from transformers import AutoModel,AutoTokenizer
import torch
import argparse
from torch.utils.data import Dataset
import json
import math
from tqdm import tqdm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt
from compute_cka import *
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from ft_datasets.dolly_dataset.dolly_dataset import get_dolly_dataset
from ft_datasets.alpaca_dataset.alpaca_dataset import get_alpaca_dataset
from ft_datasets.gsm8k.gsm8k_dataset import get_gsm8k_dataset
from ft_datasets.boolq_dataset.boolq_dataset import get_boolq_dataset
from ft_datasets.commonsenseqa.commonsenseqa_dataset import get_commonsense_dataset
from ft_datasets.openbookqa.openbookqa_dataset import get_openbookqa_dataset
from ft_datasets.openplatypus.openplatypus_dataset import get_openplatypus_dataset

from configs.datasets import alpaca_dataset as alpaca_dataset_config
from configs.datasets import dolly_dataset as dolly_dataset_config
from configs.datasets import gsm8k_dataset as gsm8k_dataset_config
from configs.datasets import commonsenseqa_dataset as commonsenseqa_dataset_config
from configs.datasets import boolq_dataset as boolq_dataset_config
from configs.datasets import openbookqa_dataset as openbookqa_dataset_config
from configs.datasets import openplatypus_dataset_math as openplatypus_math_config
from configs.datasets import openplatypus_dataset_scienceqa as openplatypus_scienceqa_config
from configs.datasets import openplatypus_dataset_airoboros as openplatypus_airoboros_config
from configs.datasets import openplatypus_dataset as openplatypus_config

class GetHookedValue:
    def __init__(self, model_name, device='cpu',checkpoint_value=None,tokneizer=None):
        self.model_name = model_name
        self.device = device
        self.activations = {}
        self.handles = []
    

        # Load the tokenizer and model
        
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.tokenizer=tokneizer

        # Set model to evaluation mode
        self.store_feature_list = []
        self.layer_output=[]
        self.model.eval()

        self.register_hooks()

    def hook_fn(self, module, input, output):
        # Store the output activations
        module_name = module.__class__.__name__
        if module_name not in self.activations:
            self.activations[module_name] = []
   
        res=output[0].detach().cpu()
        self.activations[module_name].append(res)

    def register_hooks(self):
        # Register hooks on all modules or filter as needed
        for name, module in self.model.named_modules():
            if isinstance(module, (Phi3DecoderLayer)):
                handle = module.register_forward_hook(self.hook_fn)
                self.handles.append(handle)

    def remove_hooks(self):
        # Remove all hooks to avoid memory leaks
        for handle in self.handles:
            handle.remove()

    def get_activations(self, inputs,only_input=False):
        # Clear activations dictionary
        self.activations = {}

        # if isinstance(inputs, str):
        #     inputs.to(self.model.device)
        # else:
        #     inputs = tokenizer.decode(inputs["input_ids"])
            #inputs= inputs["input_ids"].to(self.model.device)
    
        # inputs["input_ids"] = inputs["input_ids"].unsqueeze(0).to(self.model.device)
        # inputs["attention_mask"] = inputs["attention_mask"].unsqueeze(0).to(self.model.device)

        if only_input:
            if "labels" in inputs:
                labels = inputs["labels"]
                mask = labels < 0  # Create a mask for tokens with label -100
                inputs["input_ids"] = inputs["input_ids"][mask]
                inputs["attention_mask"] = inputs["attention_mask"][mask]
                del inputs["labels"]  # Remove the labels key as it's no longer needed

        else:
            if "labels" in inputs:
                del inputs["labels"]
        
        inputs["input_ids"] = inputs["input_ids"].unsqueeze(0).to(self.model.device)
        inputs["attention_mask"] = inputs["attention_mask"].unsqueeze(0).to(self.model.device)
      
        # Run the model (no gradient computation needed)
        with torch.no_grad():
            self.model(**inputs)
        

        # Now self.activations contains the activations from the forward pass
      
        for key in self.activations.keys():
            
            self.activations[key]=torch.stack(self.activations[key], dim=1).squeeze(0)[:,-1,:] # only get the last token
            self.activations[key]=self.activations[key].squeeze(1)
        return self.activations
        

    def inference(self, text_data,only_input=False):

        for data in tqdm(text_data):
            
            res=self.get_activations(data,only_input=only_input)
              
            self.layer_output.append(res['Phi3DecoderLayer'])
        
        return self.layer_output    
    

if __name__=="__main__":
    #device=torch.device( "cpu")
    device=torch.device("cuda:0")
    parser = argparse.ArgumentParser(description='  ')

    parser.add_argument('--modelname', required=True, type=str,help='on the XX model')
    parser.add_argument('--dataset', required=False,  type=str, help='Dataset Name')

    parser.add_argument('--step', required=False, default=None, type=int, help='Training step')
    parser.add_argument('--type', required=False, default="linear", type=str, help='CKA type')
    parser.add_argument('--modelname2', required=False, default=None, type=str, help='on the XX model')
    parser.add_argument('--step2', required=False, default=None, type=int, help='Training step')


    args = parser.parse_args()

    modelname=args.modelname
    datasetname=args.dataset
    step=args.step 
    compute_type=args.type
    modelname2=args.modelname2
    step2=args.step2

    tokenizer = AutoTokenizer.from_pretrained(modelname)
    # tokenizer.add_special_tokens(
    #         {
    #             "pad_token": "<PAD>",
    #         }
    #     )
    
    if "dolly" in datasetname.lower():
        dolly_dataset_config.data_path = "./ft_datasets/dolly_dataset/dolly-15k-test.jsonl"
        dataset=get_dolly_dataset(dolly_dataset_config, tokenizer, partition="whole",pad=False)
    elif "alpaca" in datasetname.lower():
        alpaca_dataset_config.data_path = "./ft_datasets/alpaca_dataset/alpaca_test.json"
        dataset=get_alpaca_dataset(alpaca_dataset_config, tokenizer, partition="whole",pad=False)
    elif "gsm8k" in datasetname.lower():
        gsm8k_dataset_config.data_path = "./ft_datasets/gsm8k/gsm8k_test.jsonl"
        dataset=get_gsm8k_dataset(gsm8k_dataset_config, tokenizer, partition="test",pad=False)
    elif "boolq" in datasetname.lower():
        boolq_dataset_config.data_path = "./ft_datasets/boolq_dataset/boolq_test.jsonl"
        dataset=get_boolq_dataset(boolq_dataset_config, tokenizer, partition="test",pad=False)
    elif "commonsenseqa" in datasetname.lower():
        commonsenseqa_dataset_config.data_path = "./ft_datasets/commonsenseqa/commonsenseqa_val.jsonl"
        dataset=get_commonsense_dataset(commonsenseqa_dataset_config, tokenizer, partition="test",pad=False)
    elif "openbookqa" in datasetname.lower():
        openbookqa_dataset_config.data_path = "./ft_datasets/openbookqa/openbookqa_test.jsonl"
        dataset=get_openbookqa_dataset(openbookqa_dataset_config, tokenizer, partition="test",pad=False)
    
    elif "openplatypus_math" in datasetname.lower():
        openplatypus_math_config.data_path = "./ft_datasets/openplatypus/MATH_PRM-800K_test.jsonl"
        dataset=get_openplatypus_dataset(openplatypus_math_config, tokenizer, partition="whole",pad=False)
    elif "openplatypus_scienceqa" in datasetname.lower():
        openplatypus_scienceqa_config.data_path = "./ft_datasets/openplatypus/scienceqa_test.jsonl"
        dataset=get_openplatypus_dataset(openplatypus_scienceqa_config, tokenizer, partition="whole",pad=False)
    elif "openplatypus_airoboros" in datasetname.lower():
        openplatypus_airoboros_config.data_path = "./ft_datasets/openplatypus/airoboros_test.jsonl"
        dataset=get_openplatypus_dataset(openplatypus_airoboros_config, tokenizer, partition="whole",pad=False)
    elif "openplatypus_full" in datasetname.lower():
        openplatypus_config.data_path = "./ft_datasets/openplatypus/Open-Platypus_test.jsonl"
        dataset=get_openplatypus_dataset(openplatypus_config, tokenizer, partition="whole",pad=False)
    
   
    # dataset = Instructiondataset(data_path=data_path, tokenizer=tokenizer,max_word=256,  pad=False, dataset_size=300, )
    # dataset = dataset.get_subset(300)

   
    Hook=GetHookedValue(modelname,device=device,checkpoint_value=step,tokneizer=tokenizer) 

    res_inference=Hook.inference(dataset,only_input=True) 
    
    
    # if modelname2 is not None:
    #     Hook2=GetHookedValue(modelname2,device=device,checkpoint_value=step2) 
    #     res_inference2=Hook2.inference(dataset)

    
    print("Get the inference results")

    res_inference=torch.stack(res_inference, dim=0)    
    layer_list =list(torch.unbind(res_inference, dim=1))
    cka_result_matrix=torch.zeros(len(layer_list), len(layer_list))


    if device.type == 'cuda':
        print('Using GPU for CKA computation')
        cuda_cka=CudaCKA(device=device)

        for i in tqdm(range(len(layer_list))):
            for j in range(len(layer_list)):
                if compute_type =="linear":
                    cka = cuda_cka.linear_CKA(layer_list[i].to(device=device), layer_list[j].to(device=device))
                elif compute_type =="kernel":
                    cka = cuda_cka.kernel_CKA(layer_list[i].to(device=device), layer_list[j].to(device=device)) 

                cka_result_matrix[i][j]=cka
    
    elif device.type == 'cpu':
        print('Using CPU for CKA computation')

        

        Cka=CKA()
        for i in tqdm(range(len(layer_list))):
            for j in range(len(layer_list)):
                cka = Cka.linear_CKA(layer_list[i], layer_list[j])
                cka_result_matrix[i][j]=cka


    print(cka_result_matrix)

    file_path=modelname.split('/')[-1]

    if step is not None:
        file_path+="/step"+str(step)

    dict_path="./model_cka/"+file_path

    if not os.path.exists(dict_path):
        os.makedirs(dict_path)

    if "dolly" in datasetname.lower():
        datasetname="dolly"
    elif "alpaca" in datasetname.lower():
        datasetname="alpaca"

    tesnor_file_path=datasetname+"_"+compute_type+"_ckares.pt"
    pic_file_path=datasetname+"_"+compute_type+"_ckares.png"

    torch.save(cka_result_matrix, os.path.join(dict_path,tesnor_file_path))

    plot_results(cka_result_matrix,save_path=os.path.join(dict_path,pic_file_path), model_name1=file_path, model_name2=file_path,normalize=True)


    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print("Finish the CKA computation on"  + modelname)
    


