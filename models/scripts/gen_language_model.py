from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering
)
from utils.utility import GLUE_TASKS

def gen_language_model(args):
    root_path = "./models/storage/language/{}/{}".format(args.model_name, args.dataset)
    # root_path += args.dataset
    config = AutoConfig.from_pretrained(root_path)
    model_generator = AutoModelForSequenceClassification if args.dataset in GLUE_TASKS else AutoModelForQuestionAnswering
    model = model_generator.from_pretrained(root_path, config=config)
    return model, config


def gen_LLM_model(args):

    from transformers import AutoTokenizer, GPT2Model, GPT2LMHeadModel
    import torch

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    return model, model.config