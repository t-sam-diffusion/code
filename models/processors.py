import torch
import torch.nn.functional as F 
from typing import Optional
from diffusers.utils import deprecate, logging
from diffusers.models.attention_processor import Attention
import math
import numpy as  np 
import matplotlib.pyplot as plt




class AttnProcessorX_2:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")


    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        # if hidden_states.shape[1] == 256:
        #     if encoder_hidden_states is not None and encoder_hidden_states.shape[1] == 77 and encoder_hidden_states.shape[0] == 2 :
        #         with open('conform_time0_4096_query_ref50.pkl', 'wb') as f:
        #             pkl.dump(hidden_states.detach().cpu(), f)
        #         exit()
                
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        # batch = 1 if self.positive_prompt else 0

        if attention_mask is not None:
            mask_shape = attention_mask.shape
            attention_mask = attention_mask.view(mask_shape[0]*mask_shape[1], 1, -1) 

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        shapes = attention_probs.shape
        attn_re = attention_probs.reshape(batch_size, attn.heads, shapes[-2], shapes[-1])

        attention_probs = attn_re.reshape(batch_size* attn.heads, shapes[-2], shapes[-1])


        hidden_states = torch.bmm(attention_probs, value)


        hidden_states = attn.batch_to_head_dim(hidden_states)

        ######_x
        if attention_probs.size()[-1] in [120,77]:
            self.attn_data_x = torch.mean(attn_re[0],dim=0) # torch.Size([1, #head, #query_flatten, 77])
            # for checking
            # print("key shape", key.shape)
            # self.key_store = key.reshape(1, 77, -1)
        ######_x


        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)


        # attn.residual_connection = True
        if attn.residual_connection:
        
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states