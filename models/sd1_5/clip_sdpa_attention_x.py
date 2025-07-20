import torch    
import torch.nn as nn
from typing import Optional, Tuple

from transformers.models.clip.modeling_clip import CLIPAttention

from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_2
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

logger = logging.get_logger(__name__)






class CLIPSdpaAttentionX(CLIPAttention):
    """
    SDPA attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `CLIPAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """
    def __init__(self, config):
        super().__init__(config)
        self.dummy = 0
        
    def get_attention_scores(
        self, query: torch.Tensor, key: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype

        # if self.upcast_attention:
        #     query = query.float()
        #     key = key.float()




        ###equivalent to head_to_batch_dim in processor:
        query = query.squeeze()
        key = key.squeeze()
        attention_mask = attention_mask.squeeze()
        # print(key.size())
        # exit() 
        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        # if self.upcast_softmax:

        #     attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    # Adapted from CLIPAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "CLIPModel is using CLIPSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not "
                "support `output_attentions=True`. Falling back to the manual attention implementation, but specifying "
                "the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can "
                'be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
            )

        # CLIP text model uses both `causal_attention_mask` and `attention_mask`
        if attention_mask is not None and causal_attention_mask is not None:
            attn_mask = attention_mask + causal_attention_mask
        elif causal_attention_mask is not None:
            attn_mask = causal_attention_mask
        else:
            attn_mask = attention_mask

        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if not is_torch_greater_or_equal_than_2_2 and query_states.device.type == "cuda" and attn_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # CLIP text model uses both `causal_attention_mask` and `attention_mask` sequentially.
        # attn_output = torch.nn.functional.scaled_dot_product_attention(
        #     query_states,
        #     key_states,
        #     value_states,
        #     attn_mask=attn_mask,
        #     dropout_p=self.dropout if self.training else 0.0,
        #     scale=self.scale,
        # )
        ##########_X
        attention_probs = self.get_attention_scores(
            query_states,
            key_states,
            attention_mask=attn_mask,
            # dropout_p=self.dropout if self.training else 0.0,
        )
        
        shapes = attention_probs.shape
        attn_re = attention_probs.reshape(bsz, self.num_heads, shapes[-2], shapes[-1])

        # attention_probs = attn_re.reshape(bsz* self.num_heads, shapes[-2], shapes[-1])

        value_states = value_states.squeeze()
        hidden_states = torch.bmm(attention_probs, value_states)


        # hidden_states = self.batch_to_head_dim(hidden_states)
        # print(self.dummy)
        ######_x
        if attention_probs.size()[-1] in [120,77]:
            # batch = 1 if self.positive_prompt else 0
            if self.dummy == 0:
                self.attn_data_x = torch.mean(attn_re,dim=1).squeeze()
                # print("store data")
        ######_x
        
        self.dummy = 1
        
        attn_output = hidden_states.to(query_states.dtype)
        attn_output = attn_output.unsqueeze(0)
        # linear proj
        # hidden_states = self.to_out[0](hidden_states)
        # # dropout
        # attn_output = self.to_out[1](hidden_states)
        ##########
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, None
    
    
