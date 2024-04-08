import torch
from utils.mac_complexity import compute_base_mac, compute_pruned_mac
from data.scripts.glue import avg_seq_length

GLUE_TASKS = [
            "stsb",
            "mrpc",
            "rte",
            "sst2",
            "qqp",
            "qnli",
            "cola",
            "mnli",
            "mnli-m",
            "mnli-mm",
]

def calculateComplexity(args, model, train_dataset, prunedProps, pruningParams={}):
    
    
    original_complexity = {
        "MAC": 1,
        "Latency": 1
    }
    
    pruned_complexity = {
        "MAC": 1,
        "Latency": 1
    }
    
    
    original_mac = compute_base_mac(args, prunedProps, skipConv=False)
    pruned_mac = compute_pruned_mac(args, prunedProps, pruningParams, skipConv=False)
    
   
    original_complexity["MAC"] = original_mac
    pruned_complexity["MAC"] = pruned_mac
    
    
    print(original_complexity, pruned_complexity)
    return original_complexity, pruned_complexity

