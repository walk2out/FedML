import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

######################################################################################
######################################################################################
def _prep_saturation_val_tensor(sat_val):
    is_scalar = not isinstance(sat_val, torch.Tensor)
    out = torch.tensor(sat_val) if is_scalar else sat_val.clone().detach()
    if not out.is_floating_point():
        out = out.to(torch.float32)
    if out.dim() == 0:
        out = out.unsqueeze(0)
    return is_scalar, out
    
def asymmetric_linear_quantization_params(num_bits, saturation_min, saturation_max,
                                          integral_zero_point=True, signed=False):
    scalar_min, sat_min = _prep_saturation_val_tensor(saturation_min)
    scalar_max, sat_max = _prep_saturation_val_tensor(saturation_max)
    is_scalar = scalar_min and scalar_max

    if scalar_max and not scalar_min:
        sat_max = sat_max.to(sat_min.device)
    elif scalar_min and not scalar_max:
        sat_min = sat_min.to(sat_max.device)

    if any(sat_min > sat_max):
        raise ValueError('saturation_min must be smaller than saturation_max')

    n = 2 ** num_bits - 1

    # Make sure 0 is in the range
    sat_min = torch.min(sat_min, torch.zeros_like(sat_min))
    sat_max = torch.max(sat_max, torch.zeros_like(sat_max))

    diff = sat_max - sat_min
    # If float values are all 0, we just want the quantized values to be 0 as well. So overriding the saturation
    # value to 'n', so the scale becomes 1
    diff[diff == 0] = n

    scale = n / diff
    zero_point = scale * sat_min
    if integral_zero_point:
        zero_point = zero_point.round()
    if signed:
        zero_point += 2 ** (num_bits - 1)
    if is_scalar:
        return scale.item(), zero_point.item()
    return scale, zero_point

def linear_quantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(scale * input - zero_point)

def asym_linear_quantize(input, b_w, c):
    if b_w > 0.0 and b_w < 32.0:
        scale = c / (2 ** (b_w - 1))
        fmin = -(2 ** (b_w - 1)) * scale
        fmax = (2 ** (b_w - 1) - 1) * scale
        
        input = input.clamp(min=fmin, max=fmax)
        input = torch.round(input / scale) * scale
    elif b_w == 1:
        tem_p = (input >= 0).float() * c
        tem_n = -(input < 0).float() * c
        input = tem_p + tem_n
    
    return input

def linear_dequantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale

class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, dequantize, inplace):
        if inplace:
            ctx.mark_dirty(input)
        output = linear_quantize(input, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None, None, None
    
class SymLinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, b_w, c):
        output = asym_linear_quantize(input, b_w, c)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None

##################################################################################################
##################################################################################################
def Read_list(filename):
    file1 = open(filename, "r")
    list_row =file1.readlines()
    list_source = []
    for i in range(len(list_row)):
        column_list = list_row[i].strip().split(" ")
        list_source.append(column_list)
    file1.close()
    return list_source

def haq_quantize_param(param_fp, b_w, c):
    out = SymLinearQuantizeSTE.apply(param_fp, b_w, c)

    return out

def find_threshold(param_fp, b_w, c_old, n=20, step=0.001):
    min_c = 0
    KL_min = 100000
    for c in np.linspace(max(c_old - step, step), c_old + step, n):
        q_weight = asym_linear_quantize(param_fp, b_w, c)
        # KL_loss = F.kl_div(param_fp, q_weight)
        KL_loss = (param_fp - q_weight).abs().sum()
        if KL_loss < KL_min:
            KL_min = KL_loss
            min_c = c
    #     print('max-mean-c_old-min_c: ', param_fp.abs().max(), param_fp.abs().mean(), c_old, min_c)
    return min_c

def haq_quantize_model(model_fp, b_w=32, mp_list=None):
    conv_idx = -1
    model_quant = deepcopy(model_fp)
    for name, module in model_quant.model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_idx = conv_idx + 1
            if mp_list is not None:
                b_w = mp_list[conv_idx]
            c_old = module.weight.data.abs().max().cpu()
            min_c = find_threshold(module.weight.data, b_w, c_old)
            module.weight.data[...] = haq_quantize_param(module.weight.data, b_w, min_c)

    return model_quant
