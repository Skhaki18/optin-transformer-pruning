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


@torch.no_grad()
def Prune(args, prunedProps, batch, model, base_layer_wise_output, base_logit_output, base_grad_output=None):
    PerNeuronMasking = -1*(torch.eye(prunedProps["inter_size"]) - 1) 
    
    globalNeuronRanking = PriorityQueue() 
    averageScaling = []
    for layer in range(prunedProps["num_layers"]):
        
        print("Layer Sample: ", layer, " / ", prunedProps["num_layers"])
        if layer==0:
            neuronRanking = PriorityQueue() 
            
        for neuron in (range(prunedProps["inter_size"])):
            then = time.time()
            torch.cuda.synchronize()
            
            maskingProps = {
                "state":"neuron",
                "layer": layer,
                "module": "intermediate-intermediate_act_fn", # This is for a pre-hook
                "mask": PerNeuronMasking[neuron]
            }
            
            if args.task_name == "fusion": maskingProps["fusion_component"] = prunedProps["fusion_component"]
            if args.task_name == "fusion" and "_m" in prunedProps["component"]:
                # must shift index by **12
                maskingProps["layer"] += 12
            
            modelObject = ModelHooking(args=args, model=model.eval(), maskProps=maskingProps)
            with torch.no_grad():
                current_logit_output, current_layer_wise_output = modelObject.forwardPass(batch)
            modelObject.purge_hooks()
            
            MMDLayerResults = 0
            KLErr = 0
            
            expFunction1 = torch.tensor(list(np.geomspace(start=1, stop=100, num=prunedProps["inter_size"]))[::-1])
            if args.loss_type == "MMD" or args.loss_type == "MMD+KL":
                
                
                if (layer == prunedProps["num_layers"]-1) and args.loss_type == "MMD":
                    print("Only Here if no KL and we are on Final Layer to generate scores!")
                    with torch.no_grad():
                        err = manifold_Distillation(args, base_layer_wise_output[-1], current_layer_wise_output[-1])
                        MMDLayerResults += err
                        
                for idx in range(len(base_layer_wise_output)):
                    if idx > layer  or ((layer == prunedProps["num_layers"]-1) and layer == idx and args.neuron_include_fin_layer_mmd):
                        with torch.no_grad():
                            err = manifold_Distillation(args, base_layer_wise_output[idx], current_layer_wise_output[idx])
                            MMDLayerResults += err
            
            
            if args.loss_type == "KL" or args.loss_type == "MMD+KL":    
                KLErr = KLDiv(base_logit_output, current_logit_output, temp=args.temp)
            
            MMDResults = 0
            MMDResults += MMDLayerResults
            
            if MMDLayerResults < KLErr: # losses are negative
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
            MMDResults *= expFunction1[neuron]
            
            MMDResults *= args.neuron_metric_decay
            
            if (args.neuron_fin_decay_bool and layer == prunedProps["num_layers"]-1):
                MMDResults *= args.neuron_fin_decay_val
                
            
            print("Err", MMDLayerResults, KLErr)
            try:
                assert KLErr <= 0 and MMDLayerResults <= 0
            except AssertionError:
                print("Non-negative MMD Result")
                print(layer, neuron, MMDResults, KLErr, MMDLayerResults, expFunction1[neuron])
                exit(1)
                
            if layer == 0:
                neuronRanking.put((MMDResults.detach().cpu(), neuron))
                
            globalNeuronRanking.put((MMDResults.detach().cpu(), layer, neuron, "neuron"))
            now = time.time()
            print("Layer Sample: ", layer, " / ", prunedProps["num_layers"], "::  Neuron Sample: ", neuron, " / ", prunedProps["inter_size"]-1, "Time: ", now-then, "Err:", MMDResults)
        
    return globalNeuronRanking, neuronRanking

            
def pruneLanguageNeurons(model, train_dataset, args, prunedProps):
    
    storage_path_cap = "./storage/{}/{}/{}/neuron_ranking_cap.pkl".format(args.task_name, args.dataset, args.model_name)
    
    storage_path_body = "./storage/{}/{}/{}/neuron_ranking_body.pkl".format(args.task_name, args.dataset, args.model_name)
    
    final_neuron_mask = torch.zeros((prunedProps["num_layers"],prunedProps["inter_size"]))
    
    prunedProps["lambda"] = args.lambda_contribution
    
    if not os.path.isfile(storage_path_body):
        
        torch.backends.cudnn.benchmark = True
        
        batch = next(iter(train_dataset))
        if args.task_name == "language":
            for k, v in batch.items():
                batch[k] = v.to("cuda", non_blocking=True)    
        # Compute Baseline Results
        model.eval()
        modelObject = ModelHooking(args=args, model=model)
        
        
        base_logit_output, base_layer_wise_output = modelObject.forwardPass(batch)
        modelObject.purge_hooks()


        globalNeuronRanking, neuronRanking = Prune(args, prunedProps, batch, model, base_layer_wise_output, base_logit_output)

        exportglobalNeuronRanking = []
        exportneuronRanking = []
        
        while not globalNeuronRanking.empty():
            exportglobalNeuronRanking.append(globalNeuronRanking.get())
        
        while not neuronRanking.empty():
            exportneuronRanking.append(neuronRanking.get())
        
        with open(storage_path_body, 'wb') as f:
            pickle.dump(exportglobalNeuronRanking, f)
            
        with open(storage_path_cap, 'wb') as f:
            pickle.dump(exportneuronRanking, f)
            
        
    else:
        
        with open(storage_path_body, 'rb') as f:
            exportglobalNeuronRanking = pickle.load(f)
            
        with open(storage_path_cap, 'rb') as f:
            exportneuronRanking = pickle.load(f)
    
    
    originalGlobal = exportglobalNeuronRanking.copy()
    return {"final_neuron_ranking":originalGlobal}