import torch
from models.scripts.gen_language_model import gen_language_model, gen_LLM_model

# Vision
# from models.scripts.gen_vision_model import gen_vision_model

def generateModel(args):
    
    model = None
    
    if args.task_name == "LLM":
        model, model_config = gen_LLM_model(args)
        
    if args.task_name == "language":
        model, model_config = gen_language_model(args)
    
    if args.task_name == "vision":
        model, model_config = gen_vision_model(args)
        
    if model is None:
        raise Exception("No Model Found !!")
    
    return model, model_config