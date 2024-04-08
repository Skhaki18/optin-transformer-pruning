import torch
from prune.language_pruning import pruneLanguageNeurons
from prune.head_pruning import pruneHead
import numpy as np
from utils.mac_complexity import get_mac_details



global Head_mean
global Neuron_mean

def pruneModel(args, model, train_dataset, model_config):
    
    prunedProps = {
        "num_att_head": model_config["num_attention_heads"],
       "inter_size": model_config["intermediate_size"],
       "hidden_size": model_config["hidden_size"],
       "num_layers":model_config["num_hidden_layers"],
       "patch_size": args.seq_len+1 
    }
    
    
    if args.task_name == "language":
        
        head_mask_results = pruneHead(model, train_dataset, args, prunedProps)
        
        # Only Prune Heads and Intermediate Neurons
        intermediate_neuron_results = pruneLanguageNeurons(model, train_dataset, args, prunedProps)
        
        
        languageProps = {
            "head_results": head_mask_results,
            "intermediate_results": intermediate_neuron_results,
            "mac_details": get_mac_details(args, prunedProps)
            
        }
        
        masks = globalRankingLanguage(args, prunedProps, languageProps)
        
        # Saving Final Masks:
        
        # storage_path_head = "./storage/{}/{}/{}/fina_head_mask_61.pt".format(args.task_name, args.dataset, args.model_name)
        
        # storage_path_neuron = "./storage/{}/{}/{}/fina_neuron_mask_61.pt".format(args.task_name, args.dataset, args.model_name)
        # torch.save(masks["head_mask"], storage_path_head)
        
        # torch.save(masks["intermediate_mask"], storage_path_neuron)
        
        
        pruningParams = {
        "head_mask":  masks["head_mask"], 
        "neuron_mask": masks["intermediate_mask"],
        }
        
    
    return pruningParams





def globalRankingLanguage(args, prunedProps, languageProps):
    head_mask = languageProps["head_results"]["final_head_ranking"]
    head_rank = [list((tensor_cpu.cpu().detach().item(), *rest)) for tensor_cpu, *rest in head_mask]
    head_rank = np.array(head_rank)
    
    neuron_mask = languageProps["intermediate_results"]["final_neuron_ranking"]
    neuron_rank = [list((tensor_cpu.cpu().detach().item(), *rest)) for tensor_cpu, *rest in neuron_mask]
    neuron_rank = np.array(neuron_rank)
    
    head_mac = languageProps["mac_details"]["head_mac"]
    neuron_mac = languageProps["mac_details"]["neuron_mac"]
    baseline_mac = languageProps["mac_details"]["base_mac"]
    
    capacity_mac = args.mac_constraint * baseline_mac
    
    
    max_importance = 0
    for num_heads in (range(1, prunedProps["num_att_head"]*prunedProps["num_layers"] + 1)):
        current_importance = 0
        
        for i in range(num_heads):
            score, _, _, _ = head_rank[i]
            current_importance += -1*float(score)
        
        count_head_mac = head_mac * (num_heads)
        remaining_mac = capacity_mac - count_head_mac
        
        num_neurons=0
        while remaining_mac >= neuron_mac and num_neurons < len(neuron_rank):
            score, neuron_layer, neuron_index, name = neuron_rank[num_neurons]
            current_importance += -1*float(score)
            num_neurons +=1 
            remaining_mac -= neuron_mac
        
        if current_importance > max_importance:
            max_importance = current_importance
            head_indicies = num_heads
            neuron_indicies = num_neurons
    
    final_head_mask = torch.zeros((prunedProps["num_layers"],prunedProps["num_att_head"]))
    final_neuron_mask = torch.zeros((prunedProps["num_layers"],prunedProps["inter_size"]))
    
    for i in range(head_indicies):
        score, head_layer, head_index, name = head_rank[i]
        final_head_mask[int(head_layer)][int(head_index)] = 1
        
    for i in range(neuron_indicies):
        score, neuron_layer, neuron_index, name = neuron_rank[i]
        final_neuron_mask[int(neuron_layer)][int(neuron_index)] = 1
    
    
    print(final_head_mask.sum(-1),final_neuron_mask.sum(-1))
    
    masks = {
        "head_mask": final_head_mask,
        "intermediate_mask": final_neuron_mask
    }
    
    return masks

def staticSizeReduction(args, model, masks):
        
    params_num = sum(p.numel() for p in model.parameters())
    
    return