import torch
from torch.nn import functional as F
import numpy as np

# Base MD Implementation
def manifold_Distillation(args, teacher, student):
    err = 0
    F_s = student
    F_t = teacher
    
    F_s = F.normalize(F_s, dim=-1)
    F_t = F.normalize(F_t, dim=-1)
    
    
    K = 768 # Subject to change based on architecture.
    bsz, patch_num, _ = F_s.shape
    
    sampler = torch.randperm(bsz * patch_num)[:K]

    f_s = F_s.reshape(bsz * patch_num, -1)[sampler]
    f_t = F_t.reshape(bsz * patch_num, -1)[sampler]

    M_s = f_s.mm(f_s.T)
    M_t = f_t.mm(f_t.T)

    M_diff = M_t - M_s
    
    if args is None:
        loss_mf_rand = (M_diff * M_diff).sum()
    elif args.aggregate == 'sum':
        loss_mf_rand = (M_diff * M_diff).sum()
    elif args.aggregate == 'mean':
        loss_mf_rand = (M_diff * M_diff).mean()
    else:
        raise Exception('aggregate not specified')
    
    err += -1*loss_mf_rand
    
    return err


# MD Applied to CNN-based Architectures
def cnn_mmd(teacher, student):
    err = 0
    F_s = student
    F_t = teacher
    
    F_s = F.normalize(F_s, dim=-1)
    F_t = F.normalize(F_t, dim=-1)
    
    
    loss_mf_rand += torch.sum((torch.mean(F_t, dim=0) - torch.mean(F_s, dim=0)))**2
    
    err += -1*loss_mf_rand
    
    return err


## Standard KL
def KLDiv(TeacherOutput,StudentOutput, temp=4):
    T = temp
    kl_div = F.kl_div(
            F.log_softmax(StudentOutput.logits, dim=1),
            F.log_softmax(TeacherOutput.logits, dim=1),
            reduction='sum',
            log_target=True
        ) * (T * T)
    
    kl_div = torch.clamp(kl_div, min=0)
    
    return -1*(kl_div)


## Alternate Patch based MD -- only for Vision
def patch_based_manifold_Distillation(teacher, student, layer):
    err = 0
    F_s = student
    F_t = teacher
    
    F_s = F.normalize(F_s, dim=-1)
    F_t = F.normalize(F_t, dim=-1)
    
    # manifold loss among different samples (inter-sample) -- directly compares along token
    f_s = F_s.permute(1, 0, 2)
    f_t = F_t.permute(1, 0, 2)

    M_s = f_s.bmm(f_s.transpose(-1, -2))
    M_t = f_t.bmm(f_t.transpose(-1, -2))

    M_diff = M_t.mean(0) - M_s.mean(0)
    loss_mf_sample = (M_diff * M_diff).sum()
    
    err += -1*loss_mf_sample
    
    return err