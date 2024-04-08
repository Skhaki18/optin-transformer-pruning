import torch
from collections import OrderedDict
import torch.nn as nn
import numpy as np
import math
from typing import Callable, Tuple

class ModelHooking():
    def __init__(self, args, model=None, maskProps=None, evalState=None):
        super(ModelHooking, self).__init__()
    
        self.layer_outputs = []
        self.layer_hooks = []
        
        self.neuron_hook = []
        self.hidden_hook = []
        self.qkv_uniform_hook = []
        self.patch_hook = []
        self.head_mask = None
        
        self.model = model.cuda()
        self.args = args 
        self.evalProps = evalState
        
        self.distill_token = args.distill_token
        
        if self.evalProps is None:
            self.maskProps = maskProps
            if self.maskProps:
                self.apply_mask()
            self.forward_hook()
        else:
            print("Applying Final mask!!!") 
            if args.task_name=="language":
                neuronHooks = self.full_neuron_mask()
            
            # For timm:
            else:
                neuronHooks = self.timm_full_neuron_mask()
            if args.task_name == "language": 
                
                self.evalProps["patch_mask"] = torch.zeros((12))
                return
            
    
    """DO NOT MODIFY: Basic Util Hook Functions"""
    
    def record_layer_output(self, name):
        """ DO NOT MODIFY: Util Function for layer record"""
        def hook(module, input, output):
            self.layer_outputs.append(output)
        return hook
    
    def heuristicPatternPatch(self):
        P = int(196 ** 0.5)
        xs = torch.linspace(-P//2, P//2, steps=P)
        ys = torch.linspace(-P//2, P//2, steps=P)
        x, y = torch.meshgrid(xs, ys)
        z = torch.sqrt(x * x + y * y)
        grid = np.array(z.view(-1))
        # print(grid)
        return grid
    
    def setHook(self, module, maskVar, patch=False):
        """ DO NOT MODIFY: Util function for Pre-Forward Masking Hook"""
        
        def localhook2(module, inputs):
            # print("in here", inputs[0].shape, inputs[1].shape)
            inputVal = inputs[0].cpu()
            sqMaskVar = maskVar.squeeze()
            
            if self.args.task_name == "vision":
                assert len(inputs) == 3
                
            K = sqMaskVar.sum()
            if inputVal.shape[1] == K or inputVal.shape[1] == K+1:
                return (inputVal.cuda(), inputs[1], inputs[2])
            
            
            
            
            merge, _ = self.bipartite_soft_matching(
                inputVal,
                int(inputVal.shape[1] - K),
                True,
                self.distill_token,
            )
            x,_ = self.merge_wavg(merge, inputVal,None)
            return (x.cuda(), inputs[1], inputs[2])
            
        
        def localhook(module, inputs):
            return (inputs[0] * maskVar.cuda(),)
        
        if patch:
             hookHandle = module.register_forward_pre_hook(localhook2)
        else:
            hookHandle = module.register_forward_pre_hook(localhook)
        return hookHandle
    
    def apply_mask(self):
        """Setting Mask for pruning phase"""
        if self.maskProps['state'] == "neuron":
            self.forward_neuron_pre_hook()
        
        elif self.maskProps['state'] == "hidden_out":
            self.forward_hidden_pre_hook_out()
        
        elif self.maskProps['state'] == "hidden_att":
            self.forward_hidden_pre_hook_att()
        
        elif self.maskProps['state'] == "head":
            self.head_mask = self.maskProps["mask"]
            
        elif self.maskProps['state'] == "qkv":
            self.forward_uniform_qkv_pre_hook()
            
        elif self.maskProps['state'] == "patch":
            self.forward_patch_pre_hook()
            
        return
    
    def purge_hooks(self):
        """Deleting Hooks"""
        for handle in self.layer_hooks:
            handle.remove()
        for handle in self.neuron_hook:
            handle.remove()
        for handle in self.hidden_hook:
            handle.remove()
        
        for handle in self.qkv_uniform_hook:
            handle.remove()
            
        for handle in self.patch_hook:
            handle.remove()
            
        self.layer_outputs = []
        
        self.neuron_hook = []
        self.hidden_hook = []
        self.qkv_uniform_hook = []
        self.layer_hooks = []
        return
        
    def return_model(self):
        return self.model
    
    
    def get_backbone(self, model):
        model_type = model.base_model_prefix
        backbone = getattr(model, model_type)
        return backbone

    def get_encoder(self, model):
        backbone = self.get_backbone(model)
        encoder = backbone.encoder
        return encoder


    def get_layers(self, model):
        encoder = self.get_encoder(model)
        layers = encoder.layer
        return layers
    
    def get_LLM_layers(self, model):
        model_type = model.base_model_prefix
        backbone = getattr(model, model_type)
        layers = backbone.h
        return layers

    ###############################################
    
    ###############################################
    
    """Pruning Phase Hooks -- MUST USE PRE-HOOKS!"""  
           
    def forward_neuron_pre_hook(self):
        """DO NOT MODIFY: Setting Mask for pruning phase"""
        if self.args.task_name == "language":
            module = self.get_layers(self.model)[self.maskProps["layer"]].intermediate.intermediate_act_fn
        
        elif self.args.task_name == "LLM":
            module = self.get_LLM_layers(self.model)[self.maskProps["layer"]].mlp.c_proj # Layer after: c_fc.weight [768,3072] c_proj [3072,768]
            
        elif self.args.task_name == "vision":
            module = self.get_layers(self.model)[self.maskProps["layer"]].intermediate.intermediate_act_fn
        elif self.args.task_name == "fusion":
            module = self.get_layers(self.model)[self.maskProps["layer"]].intermediate.intermediate_act_fn
        mask = self.maskProps["mask"]
        hook = lambda _, inputs: (inputs[0]*mask.cuda(),) #32x77x3072
        hookHandle = module.register_forward_pre_hook(hook)
        self.neuron_hook.append(hookHandle)
        return
    
    
    def forward_patch_pre_hook(self):
        """DO NOT MODIFY: Setting Mask for pruning phase"""
        
        if self.args.task_name in ["language", "LLM"]:
            return
            
        elif self.args.task_name == "vision":
            module = self.get_layers(self.model)[self.maskProps["layer"]] # Pre-Hooking Entire Layer
            
        mask = self.maskProps["mask"]
        # hook = lambda _, inputs: (inputs[0]*mask.cuda().unsqueeze(-1).unsqueeze(0),inputs[1], inputs[2])
        hook = lambda _, inputs: (inputs[0][:, mask==1, :],inputs[1], inputs[2])  
        hookHandle = module.register_forward_pre_hook(hook)
        self.hidden_hook.append(hookHandle)
        return
    
    ###############################################
    
    ###############################################
    
    """Evaluation Phase Hooks -- MUST USE PRE-HOOKS!"""
    
    def full_neuron_mask(self):
        """DO NOT MODIFY: Applies the finalized Mask for evaluation Phase"""
        hooksHandler  = []
        count = 0
        fusion = True if (self.args.task_name=="fusion") else False
        for name, module in self.model.named_modules():
            testing = True if ('intermediate' in name and 'intermediate_act_fn' in name) else False
            if ((testing and not fusion) or (testing and fusion and self.maskProps["fusion_component"] in module)):
                mask = self.evalProps["neuron_mask"][count]
                mask.requires_grad_()
                hookHandle = self.setHook(module,mask)
                hooksHandler.append(hookHandle)
                count+=1
            
            if (self.args.task_name == "LLM" and "mlp.c_proj" in name):
                mask = self.evalProps["neuron_mask"][count]
                mask.requires_grad_()
                hookHandle = self.setHook(module,mask)
                hooksHandler.append(hookHandle)
                count+=1
                            
        return hooksHandler
    
    
    def hookHere(self, name, mask):
        """ DO NOT MODIFY: Util Function for layer record"""
        def hook(module, input, output):
            # print(output.shape, mask.shape)
            return output*mask.cuda()
        return hook
    
    def timm_full_neuron_mask(self):
        hooksHandler  = []
        count = 0
        for name, module in self.model.named_modules():
            if "mlp.fc1" in name:
                print("Applied")
                mask = self.evalProps["neuron_mask"][count]
                # hook = lambda _, inputs, outputs: (outputs[0]*mask.cuda(),) #32x77x3072
                hookHandle = module.register_forward_hook(self.hookHere(name, mask))
                hooksHandler.append(hookHandle)
                count+=1
        return hooksHandler
    
    
    
    def full_hidden_out_mask(self):
        """DO NOT MODIFY: Applies the finalized Mask for evaluation Phase"""
        hooksHandler  = []
        count = 0
        fusion = True if (self.args.task_name=="fusion") else False
        
        for name, module in self.model.named_modules():
            if ('layernorm_before' in name and (self.args.task_name == "vision" or (fusion and self.maskProps["fusion_component"] in module))) \
                or ('output' in name and 'LayerNorm' in name and 'attention'not in name and (self.args.task_name == "language" or (fusion and self.maskProps["fusion_component"] in module))):
                    
                hookHandle = self.setHook(module,self.evalProps["hidden_output_mask"][count])
                hooksHandler.append(hookHandle)
                count+=1
                
        return hooksHandler
    
   
        
    def patch_mask(self):
        """DO NOT MODIFY: Applies the finalized Mask for evaluation Phase"""
        hooksHandler  = []
        count = 0
        
        # self.process_patch_mask()
        
        fusion = True if (self.args.task_name=="fusion") else False
        base_mask = None
        for name, module in self.model.named_modules():
            # print(name)
            
            # print(name)
            if name == "vit.encoder.layer.{}".format(count) or name == "bert.encoder.layer.{}".format(count) :
                if count == 0: 
                    count+=1
                    continue
                
                base_mask = self.evalProps["patch_mask"][count]
                
                # print(base_mask.sum())#, self.evalProps["patch_mask"][count].sum())
                mask = base_mask.unsqueeze(-1).unsqueeze(0)
                # mask.requires_grad_()
                hookHandle = self.setHook(module,mask, patch=True)
                hooksHandler.append(hookHandle)
                count+=1
                
        return hooksHandler
        
    ###############################################
    
    ###############################################
    
    """Embedding Selection & Forward for comparisons -- MUST USE POST-HOOKS!"""
    
    def forward_hook(self):
        """ Choose the position to compare embeddings"""
        for name, module in self.model.named_modules():
            # print(name)
            if ((self.args.task_name == "vision" or (self.args.task_name == "fusion" and "visual" in self.maskProps["fusion_component"]) )and self.args.embedding_choice in name and 'attention' not in name) \
                or ((self.args.task_name == "language" or (self.args.task_name == "fusion" and "text" in self.maskProps["fusion_component"])) and 'output' in name and 'dense' in name and 'attention' not in name) \
                    or (self.args.task_name == "LLM"  and ".mlp.c_fc" in name):
                    
            # if (self.args.task_name == "vision" and 'layernorm_after' in name and 'attention' not in name) \
                # or (self.args.task_name == "language" and 'output' in name and 'LayerNorm' in name and 'attention' not in name):
                
                hookHandle = module.register_forward_hook(self.record_layer_output(name))
                self.layer_hooks.append(hookHandle)
                  
    def forwardPass(self,batch):
        
        # print(batch)
        
        if self.head_mask is not None:
            with torch.no_grad():
                if self.args.task_name == "fusion":
                    logit_output = self.model(batch, head_mask={self.maskProps["fusion_component"]:self.head_mask})
                else:
                    logit_output = self.model(head_mask=self.head_mask.cuda(), **batch)
        else: 
            with torch.no_grad():
                logit_output = self.model(**batch)
        layer_wise_output = self.layer_outputs
        
        return logit_output, layer_wise_output
        
