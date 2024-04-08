import torch

# Language
from data.scripts.glue import glue_dataloader
from transformers import AutoTokenizer
from utils.utility import GLUE_TASKS

# Vision
# from data.scripts.cifar import cifar10Dataset, cifar100Dataset
# from data.scripts.imagenet import imageNetDataset
# from data.scripts.cityscapes import cityscapesDataset

from datasets import load_dataset


def generateDataset(args):
    
    train_dataset, val_dataset = None, None
    if args.task_name == "vision": # Vision Datasets
        print("---Vision---")
        
        if args.dataset == "ImageNet":
            train_loader, val_loader = imageNetDataset(args)
            
        elif args.dataset == "Cifar10":
            train_loader, val_loader = cifar10Dataset(args)
            
        elif args.dataset == "Cifar100":
            train_loader, val_loader = cifar100Dataset(args)
            
        elif args.dataset == "Cityscapes":
            
            train_loader, val_loader = cityscapesDataset(args)

    elif args.task_name == "language": # Language Datasets
        print("---Language---")
        if args.dataset in GLUE_TASKS:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            
            train_loader = glue_dataloader(args.dataset, tokenizer, training=True, batch_size=args.batch_size)
            val_loader = glue_dataloader(args.dataset, tokenizer, training=False, batch_size=args.timer_batch_size)
    
    elif train_dataset == None or val_dataset == None:
        raise Exception("Sorry, no matching dataset was found")
    
    return train_loader, val_loader, args