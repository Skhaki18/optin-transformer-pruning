from utils.utility import GLUE_TASKS
from evals.language.eval_glue import eval_glue_acc

def evalModel(args, model, train_dataset, val_dataset, pruningParams, prunedProps):
    
    baselinePerformance, finalPerformance = 1,1
    if args.task_name == "vision": # Vision Datasets
        print("---Vision---")
        
        if "seg" in args.model_name: # Segmentation Model
            print("Segmentation Model")
            # baselinePerformance, finalPerformance = reRouteSegment(args, model, train_dataset, val_dataset, pruningParams, prunedProps)
        
        if args.fine_tune == True: # Transfer Learning on CIFAR10 & 100 
            finalPerformance = train_acc(args, model, pruningParams=pruningParams)
        
        else:
            
            finalPerformance = eval_imagenet_acc(args, model, val_dataset, args.dataset, pruningParams=pruningParams)
        
    elif args.task_name == "language": # Language Datasets
        print("---Language---")
        if args.dataset in GLUE_TASKS:
            finalPerformance = eval_glue_acc(args, model, val_dataset, args.dataset, pruningParams=pruningParams, prunedProps=prunedProps)
            
    return None, finalPerformance

    
    