import torch

def mac_per_head(
    seq_len,
    hidden_size,
    attention_head_size,
):
    per_head_qkv = lambda seq_len: 3 * seq_len * hidden_size * attention_head_size
    per_head_attn = lambda seq_len: 2 * seq_len * seq_len * attention_head_size
    per_head_output = lambda seq_len: seq_len * attention_head_size * hidden_size
    mac = per_head_qkv(seq_len) + per_head_attn(seq_len) + per_head_output(seq_len)
    return mac


def mac_per_neuron(seq_len, hidden_size):
    return 2 * seq_len * hidden_size






def compute_mac(
    head_mask,
    neuron_mask,
    patch_mask,
    seq_len,
    hidden_size,
    attention_head_size
):
    
    mac = 0.0
    num_layers = len(head_mask)
    for layer in range(num_layers):
        
        current_num_heads = head_mask[layer]
        current_num_neurons = neuron_mask[layer]
        
        
        current_num_patch = patch_mask[layer]
        attention_mac = current_num_heads * mac_per_head(current_num_patch, hidden_size, attention_head_size)
        
        if layer == num_layers -1:
            current_num_patch = patch_mask[num_layers-1]
            ffn_mac = current_num_neurons * mac_per_neuron(current_num_patch, hidden_size)
        else:
            ffn_mac = current_num_neurons * mac_per_neuron(current_num_patch, hidden_size)
         
        mac += attention_mac + ffn_mac
    return mac


def compute_base_mac(args, prunedProps, skipConv):
    
    head_mask = [prunedProps["num_att_head"]] * prunedProps["num_layers"]
    neuron_mask = [prunedProps["inter_size"]] * prunedProps["num_layers"]
    seq_length = prunedProps["patch_size"] - 1
    patch_mask = [seq_length] * prunedProps["num_layers"]
    
    
    attention_head_size = int(prunedProps["hidden_size"] / prunedProps["num_att_head"])
    
    original_mac = compute_mac(
        head_mask,
        neuron_mask,
        patch_mask,
        seq_length,
        prunedProps["hidden_size"],
        attention_head_size
    )
    if not skipConv and args.task_name == "vision":
        original_mac += query_conv_mac(args)
    return original_mac


def compute_pruned_mac(args, prunedProps, pruningParams, skipConv):
    
    head_mask = pruningParams["head_mask"].sum(-1)
    neuron_mask = pruningParams["neuron_mask"].sum(-1)
    seq_length = prunedProps["patch_size"] - 1
    
    
    if args.task_name == "vision":
        patch_mask = [i.sum()-1 for i in pruningParams["patch_mask"]]
    else:
        patch_mask = [seq_length] * prunedProps["num_layers"]
    
    
    attention_head_size = int(prunedProps["hidden_size"] / prunedProps["num_att_head"])
    
    pruned_mac = compute_mac(
        head_mask,
        neuron_mask,
        patch_mask,
        seq_length,
        prunedProps["hidden_size"],
        attention_head_size
    )
    
    if not skipConv and args.task_name == "vision":
        pruned_mac += query_conv_mac(args)
    return pruned_mac.item()



def compute_patch_mac(args, prunedProps, mac_details):
    """Computes the effect of patch removal by layer"""
    
    # Compute Base Layerwise Mac:
    layerwise_mac = mac_details["head_mac"] + mac_details["neuron_mac"]
    
    reduced_head_mac = mac_per_head(prunedProps["patch_size"] - 1 -1, 
                                 prunedProps["hidden_size"], 
                                 int(prunedProps["hidden_size"] / prunedProps["num_att_head"]))
    reduced_neuron_mac = mac_per_neuron(prunedProps["patch_size"] - 1 - 1, prunedProps["hidden_size"])
    
    reduced_layerwise_mac = reduced_head_mac + reduced_neuron_mac
    
    total_mac_reduction_per_layer = layerwise_mac - reduced_layerwise_mac
    
    patch_mac = []
    for layer in range(1, prunedProps["num_layers"]):
        patch_mac.append((prunedProps["num_layers"]-layer)*total_mac_reduction_per_layer)
        
    final_layer_reduction = mac_details["neuron_mac"] - reduced_neuron_mac
    patch_mac.append(final_layer_reduction)
    
    return patch_mac
    

def query_conv_mac(args):

    if 'tiny' in args.model_name or 'small' in args.model_name:
        output_channels = 384
    elif 'base' in args.model_name:
        output_channels = 768
        
    elif 'large' in args.model_name:
        output_channels = 1024
        
    input_channels = 3
    kernel_size = (16, 16)
    stride = (16, 16)
    input_height = 224
    input_width = 224

    output_height = (input_height - kernel_size[0]) // stride[0] + 1
    output_width = (input_width - kernel_size[1]) // stride[1] + 1
    mac_count = input_channels * output_channels * output_height * output_width * kernel_size[0] * kernel_size[1]
    
    return mac_count

def get_mac_details(args, prunedProps):
    
    
    mac_details = {
        "base_mac": compute_base_mac(args, prunedProps, skipConv=True),
        "head_mac": mac_per_head(prunedProps["patch_size"] - 1, 
                                 prunedProps["hidden_size"], 
                                 int(prunedProps["hidden_size"] / prunedProps["num_att_head"])),
        "neuron_mac": mac_per_neuron(prunedProps["patch_size"] - 1, prunedProps["hidden_size"]),
    }
    
    if args.task_name == "vision":
        patch_mac = compute_patch_mac(args, prunedProps, mac_details)
        
        mac_details["patch_mac"] = patch_mac
        
        
    return mac_details
