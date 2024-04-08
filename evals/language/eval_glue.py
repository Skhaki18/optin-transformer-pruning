import torch
from datasets import load_metric
from data.scripts.glue import target_dev_metric
from utils._hooks import ModelHooking
from tqdm import tqdm
from data.scripts.glue import avg_seq_length

@torch.no_grad()
def eval_glue_acc(args, model, val_dataloader, task_name, pruningParams=None, prunedProps=None):
    IS_STSB = model.num_labels == 1
    metric = load_metric("glue", task_name)
    earlyStop = False
    
    if pruningParams != None:
        pruned = True
        base_model = model
        
        prunedModelObject = ModelHooking(args=args, model=model, maskProps=None, evalState=pruningParams)
        model = prunedModelObject.return_model()
        mask = pruningParams["head_mask"].cuda()
    else:
        pruned = False
        seq_len = avg_seq_length(args.dataset)
        pruningParams = {
        "head_mask": torch.ones((prunedProps["num_layers"], prunedProps["num_att_head"])),
        "neuron_mask": torch.ones((prunedProps["num_layers"], prunedProps["inter_size"])),
        "patch_mask": torch.ones((prunedProps["num_layers"], int(seq_len+1)))
        }
        mask = pruningParams["head_mask"].cuda()
        
    model.eval()
    model.cuda()
    
    
    
    for idx, batch in tqdm(enumerate(val_dataloader)):
        for k, v in batch.items():
            batch[k] = v.to("cuda", non_blocking=True)
        outputs = model(head_mask = mask,**batch)
        if IS_STSB:
            predictions = outputs.logits.squeeze()
        else:
            predictions = outputs.logits.argmax(dim=-1)
        metric.add_batch(
            predictions=predictions,
            references=batch["labels"],
        )

    eval_results = metric.compute()
    target_metric = target_dev_metric(task_name)
    accuracy = eval_results[target_metric]
    return accuracy