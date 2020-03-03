
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def glore(self,x,layers):
    '''
    x: input tensor
    layers: predefined 5 bn_ac_conv layers, use _make_glore in the constructor.
    '''

    #assume channel first
    num_in = x.size(1)

    #unpack
    num_mid = num_in //4
    num_state = int(2*num_mid)
    num_node = num_mid
    bn_ac_conv1, bn_ac_conv2, bn_ac_conv3, bn_ac_conv4, bn_ac_conv5 = layers

    x_state = bn_ac_conv1(x)
    x_state_reshaped = x_state.view(x_state.size(0),x_state.size(1),-1)

    x_proj = bn_ac_conv2(x)
    x_proj_reshaped = x_proj.view(x_proj.size(0),x_proj.size(1),-1)

    x_rproj_reshaped = x_proj_reshaped

    x_n_state = torch.bmm(x_state_reshaped,x_proj_reshaped.transpose(2,1))

    x_n_rel = x_n_state

    x_n_rel = bn_ac_conv3(x_n_rel.transpose(2,1)).transpose(2,1)

    x_n_rel += x_n_state

    x_n_rel = bn_ac_conv4(x_n_rel)

    x_n_state_new = x_n_rel

    x_out = torch.bmm(x_n_state_new,x_rproj_reshaped).view(x_state.size())

    x_out = bn_ac_conv5(x_out)

    out = x + x_out

    return x
    
