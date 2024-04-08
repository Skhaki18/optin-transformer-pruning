import torch
from prune.loss_components import KLDiv, manifold_Distillation
from queue import PriorityQueue
from torch.nn import functional as F
from tqdm import tqdm
import time
from utils._hooks import ModelHooking
import numpy as np
import os
import pickle

def least_square_wrapper(original_outputs, pruned_outputs, pruning_mask):
    
    for layer in range(len(original_outputs)):
        zeroIndices = (pruning_mask[layer] == 0).nonzero()
        flatten_orginal = original_outputs[layer].mean(0).mean(0)
        flatten_pruned = pruned_outputs[layer].mean(0).mean(0)
        
        reconstruction_error = flatten_pruned - flatten_orginal
        reconstruction_error[zeroIndices] = 0
        
        newMask = pruning_mask[layer].repeat(768,1)
        print(newMask.shape)
        
        mask_params, _, _, _ = np.linalg.lstsq(newMask.detach().cpu().numpy(), reconstruction_error.detach().cpu().numpy(), rcond=None)
        
        pruning_mask[layer] = torch.tensor(mask_params).mean(0) * pruning_mask[layer]

        
    print(pruning_mask)
    return pruning_mask


@torch.no_grad()
def Prune(args, prunedProps,FullHeadMasking, batch, model, base_layer_wise_output, base_logit_output, base_grad_output=None):
    PerHeadMasking = -1*(torch.eye(prunedProps["num_att_head"]) - 1) 
    
    globalHeadRanking = PriorityQueue() 
    importance_scores = torch.zeros((prunedProps["num_layers"],prunedProps["num_att_head"]))
    averageScaling = []
    for layer in range(prunedProps["num_layers"]):
        
        print("Layer Sample: ", layer, " / ", prunedProps["num_layers"])
        if layer==0:
            headRanking = PriorityQueue() 
        for head in (range(prunedProps["num_att_head"])):
            then = time.time()
            torch.cuda.synchronize()
            
            FullHeadMasking[layer] = PerHeadMasking[head]
            
            maskingProps = {
                "state":"head",
                "layer": layer,
                "module": "output-LayerNorm", # "intermediate-dense" This is for a pre-hook
                "mask": FullHeadMasking
            }
            
            if args.task_name == "fusion": maskingProps["fusion_component"] = prunedProps["fusion_component"]
            if args.task_name == "fusion" and "_m" in prunedProps["fusion_component"]:
                # must shift index by **12
                maskingProps["layer"] += 12
            
            modelObject = ModelHooking(args=args, model=model.eval(), maskProps=maskingProps)
            with torch.no_grad():
                current_logit_output, current_layer_wise_output = modelObject.forwardPass(batch)
            modelObject.purge_hooks()
            
            MMDLayerResults = 0
            KLErr = 0
            expFunction1 = torch.tensor(list(np.geomspace(start=1, stop=10, num=prunedProps["num_att_head"]))[::-1])
            
            print(base_layer_wise_output[0].shape)
            if args.loss_type == "MMD" or args.loss_type == "MMD+KL":
                
                if (layer == prunedProps["num_layers"]-1) and args.loss_type == "MMD":
                    print("Only Here if no KL and we are on Final Layer to generate scores!")
                    print(len(base_layer_wise_output), len(current_layer_wise_output))
                    with torch.no_grad():
                        err = manifold_Distillation(args, base_layer_wise_output[-1], current_layer_wise_output[-1])
                        MMDLayerResults += err
                
                for idx in range(len(base_layer_wise_output)):
                    if idx > layer or ((layer == prunedProps["num_layers"]-1) and layer == idx and args.head_include_fin_layer_mmd):
                        with torch.no_grad():
                            
                            err = manifold_Distillation(args, base_layer_wise_output[idx], current_layer_wise_output[idx])
                            MMDLayerResults += err
            
            if args.loss_type == "KL" or args.loss_type == "MMD+KL":    
                KLErr = KLDiv(base_logit_output, current_logit_output, temp=args.temp)
            
            
            MMDResults = 0
            MMDResults += MMDLayerResults
            if MMDLayerResults < KLErr:
                try:
                    ratio = np.log10(-1*MMDLayerResults.detach().cpu().item()) - np.log10(-1*KLErr.detach().cpu().item())
                    targetRatio = np.log10(prunedProps["lambda"])
                    scaling = 10**int(ratio - targetRatio)
                    averageScaling.append(scaling)
                        
                    KLErr *= scaling
                except:
                    meanScaling = np.mean(averageScaling)
                    if np.isnan(meanScaling): meanScaling = 1
                    KLErr *= meanScaling
                    pass
            else:
                meanScaling = np.mean(averageScaling)
                if np.isnan(meanScaling): meanScaling = 1
                KLErr *= meanScaling
            
                
            MMDResults += KLErr
            
            if (args.head_fin_decay_bool and layer == prunedProps["num_layers"]-1):
                MMDResults *= args.head_fin_decay_val
            
            if args.task_name == "language":
                MMDResults *= expFunction1[head]
            
            
            MMDResults *= args.head_metric_decay
            print("Err", MMDLayerResults, KLErr, MMDResults)
            
            try:
                assert MMDResults <= 0
            except AssertionError:
                print("Non-negative MMD Result")
                print(layer, head, MMDResults, KLErr, MMDLayerResults)
                exit(1)
                
            importance_scores[layer][head] = MMDResults.detach().cpu()
            if layer == 0:
                headRanking.put((MMDResults.detach().cpu(), head))
            globalHeadRanking.put((MMDResults.detach().cpu(), layer, head, "head"))
            now = time.time()
            print("Layer Sample: ", layer, " / ", prunedProps["num_layers"], "::  Head Sample: ", head, " / ", prunedProps["num_att_head"]-1, "Time: ", now-then, "Err:", MMDResults.item())
    
    return globalHeadRanking, headRanking

def singlePass(FullHeadMasking, model, batch):
    maskingProps = {
        "state":"head",
        "layer": -1,
        "module": "output-LayerNorm", # "intermediate-dense" This is for a pre-hook
        "mask": FullHeadMasking
    }
    
    modelObject = ModelHooking(args=args, model=model.eval(), maskProps=maskingProps)
    with torch.no_grad():
        current_logit_output, current_layer_wise_output = modelObject.forwardPass(batch)
    modelObject.purge_hooks()
    return current_logit_output, current_layer_wise_output
            
def pruneHead(model, train_dataset, args, prunedProps):
    
    storage_path_cap = "./storage/{}/{}/{}/head_ranking_cap.pkl".format(args.task_name, args.dataset, args.model_name)
    
    storage_path_body = "./storage/{}/{}/{}/head_ranking_body.pkl".format(args.task_name, args.dataset, args.model_name)
    
    final_head_mask = torch.zeros((prunedProps["num_layers"],prunedProps["num_att_head"]))
    
    prunedProps["lambda"] = args.lambda_contribution
    
    if not os.path.isfile(storage_path_body):
        try: os.makedirs("./storage/{}/{}/{}/".format(args.task_name, args.dataset, args.model_name))
        except: pass
    
        FullHeadMasking = torch.ones((prunedProps["num_layers"],prunedProps["num_att_head"]))
        
        torch.backends.cudnn.benchmark = True
        
        batch = next(iter(train_dataset))
        
        if args.task_name == "language":
            for k, v in batch.items():
                batch[k] = v.to("cuda", non_blocking=True)  
                
        elif args.task_name == "vision":
            (batch_x, batch_y) = batch
            mappingBatch = {}
            mappingBatch["pixel_values"] = batch_x.to("cuda", non_blocking=True)
            mappingBatch["labels"] = batch_y.to("cuda", non_blocking=True)  
            batch = mappingBatch
            
        # Compute Baseline Results
        model.eval()
        maskingProps = {
            "state":"head",
            "layer": None,
            "module": "output-LayerNorm", # "intermediate-dense" This is for a pre-hook
            "mask": FullHeadMasking
        }
        
        
        if args.task_name == "fusion": maskingProps["fusion_component"] = prunedProps["fusion_component"]
        if args.task_name == "fusion" and "_m" in prunedProps["fusion_component"]:
            # must shift index by **12
            maskingProps["layer"] += 12
        modelObject = ModelHooking(args=args, model=model, maskProps=maskingProps)
        
        
        base_logit_output, base_layer_wise_output = modelObject.forwardPass(batch)
        modelObject.purge_hooks()
        
       
        print(model)
        globalHeadRanking, headRanking = Prune(args, prunedProps,FullHeadMasking, batch, model, base_layer_wise_output, base_logit_output) #base_grad_output

        exportglobalHeadRanking = []
        exportheadRanking = []
        
        while not globalHeadRanking.empty():
            exportglobalHeadRanking.append(globalHeadRanking.get())
        
        while not headRanking.empty():
            exportheadRanking.append(headRanking.get())
        
        with open(storage_path_body, 'wb') as f:
            pickle.dump(exportglobalHeadRanking, f)
            
        with open(storage_path_cap, 'wb') as f:
            pickle.dump(exportheadRanking, f)
            
        
    else:
        
        with open(storage_path_body, 'rb') as f:
            exportglobalHeadRanking = pickle.load(f)
            
        with open(storage_path_cap, 'rb') as f:
            exportheadRanking = pickle.load(f)
    
    
    originalGlobal = exportglobalHeadRanking.copy()

    return {"final_head_ranking":originalGlobal}
     
