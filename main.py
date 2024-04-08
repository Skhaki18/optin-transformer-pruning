import argparse
import logging
import yaml
import os
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from datetime import datetime
import json
from transformers import set_seed
from tqdm import tqdm
import timm


from data.scripts.gen_dataset import generateDataset
from models.scripts.gen_model import generateModel
from prune.main_prune import pruneModel
from utils.utility import calculateComplexity
from data.scripts.glue import avg_seq_length

from evals.gen_eval import evalModel


parser = argparse.ArgumentParser()

parser.add_argument("--config", required=True, help="YAML configuration file")

def write_json(new_data, filename, args):
    if args.logging == False: return
    with open(filename,'r+') as file:
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data.update(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 2)
    return

def main():
    args = parser.parse_args()
    # Load the YAML configuration file
    with open(args.config, "r") as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)
        
    args = argparse.Namespace(**yaml_config)

    set_seed(args.seed)

    model, model_config = generateModel(args)
    model_config = model_config if type(model_config) is dict else vars(model_config)
    
    
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    path = './logs/{}-{}-{}-{}.json'.format(args.task_name, args.dataset, args.model_name, dt_string)
    if args.logging == True:
        with open(path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    args.model_config = model_config
    train_dataset, val_dataset, args = generateDataset(args)
    
    
    
    seq_len = avg_seq_length(args.dataset)
    args.seq_len = seq_len
    
    prunedProps = {
        "num_att_head": model_config["num_attention_heads"],
       "inter_size": model_config["intermediate_size"],
       "hidden_size": model_config["hidden_size"],
       "num_layers":model_config["num_hidden_layers"],
       "patch_size": seq_len+1
    }
    
    print(prunedProps)
    pruningParams = {
       "head_mask": torch.ones((prunedProps["num_layers"], prunedProps["num_att_head"])),
       "neuron_mask": torch.ones((prunedProps["num_layers"], prunedProps["inter_size"])),
       "patch_mask": torch.ones((prunedProps["num_layers"], int(seq_len+1)))
    }
    
    baselineComplexity, prunedComplexity = calculateComplexity(args, model, train_dataset, prunedProps, pruningParams)
    
    print("BASELINE COMPLEXITY: ", baselineComplexity, "PRUNED COMPLEXITY: ", prunedComplexity)
    
    pruningParams = pruneModel(args, model, train_dataset, model_config)
    
    baselineComplexity, prunedComplexity = calculateComplexity(args, model, train_dataset, prunedProps, pruningParams)
    
        
    print("BASELINE COMPLEXITY: ", baselineComplexity, "PRUNED COMPLEXITY: ", prunedComplexity)
    
    flopReductionAmmount = 100-(prunedComplexity["MAC"]/ baselineComplexity["MAC"] * 100.0)

    print("FLOP Reduction by:{}".format(flopReductionAmmount))
    
    args.flopReductionAmmount = flopReductionAmmount
    
    baselinePerformance, finalPerformance = evalModel(args, model, train_dataset, val_dataset, pruningParams, prunedProps)
    
    performanceMetrics = {
        "Performance":{
            "Baseline": baselinePerformance,
            "Pruned": finalPerformance,
        }
    }
    
    write_json(performanceMetrics, path, args)
    
    baselineMetrics = {
        "Baseline":{
            "Baseline-MAC": baselineComplexity["MAC"],
            "Baseline-Latency":baselineComplexity["Latency"]
        }
    }
    write_json(baselineMetrics, path,args)
    
    prunedMetrics = {
        "Pruned":{
            "Pruned-MAC": prunedComplexity["MAC"],
            "Pruned-Latency":prunedComplexity["Latency"]
        }
    }
    write_json(prunedMetrics, path, args)
    
    finalMetrics = {
        "Final":{
            "MAC": prunedComplexity["MAC"] / baselineComplexity["MAC"] * 100.0,
            "Latency":prunedComplexity["Latency"] / baselineComplexity["Latency"] * 100.0
        }    
    }
    write_json(finalMetrics, path, args)
    
    print("Orig Model Perf:{}, Pruned Model Perf: {}".format(baselinePerformance, finalPerformance))
    print("FLOP Reduction by::{}".format(100-(prunedComplexity["MAC"]/ baselineComplexity["MAC"] * 100.0)))
    print("FLOP Percentage :{}".format(prunedComplexity["MAC"]/ baselineComplexity["MAC"] * 100.0))
    
    
    return
    
if __name__ == "__main__":
    main()

    
    
    
    
    
