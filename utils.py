import torch
import os
from PIL import Image
import numpy as np
import json
from models.sd1_5.clip_sdpa_attention_x import CLIPSdpaAttentionX
from models.sd1_5.storage_sd1_5 import  AttnFetchSDX_3
from models.sd1_5.pipeline_stable_diffusion_x_2 import StableDiffusionPipelineX_2
from dataclasses import dataclass
from typing import Optional, List, Tuple
from transformers import CLIPTokenizer



@dataclass
class LatentOptConfig:
    guidance_scale: float = 7.5
    max_iter_to_alter: int = 30
    iterative_refinement_steps: List[int] = (0, 10, 20)
    refinement_steps: int = 20
    scale_factor: int = 5
    attn_res: Tuple[int, int] = (16, 16)
    do_smoothing: bool = True
    smoothing_sigma: float = 0.5
    smoothing_kernel_size: int = 3
    temperature: float = 0.5
    softmax_normalize: bool = False
    softmax_normalize_attention_maps: bool = False
    add_previous_attention_maps: bool = True
    previous_attention_map_anchor_step: Optional[int] = None
    loss_fn: str = "ntxent"
    k: int = 3
    attn_like_loss: Optional[str] = None
    cos1_or_cos2: Optional[str] = None
    loss_type: Optional[str] = None
    row_weight: Optional[float] = None
    update_latent: bool = True
    cos_sim_dir:str = './cross_attn'

    def to_dict(self):
        return self.__dict__



def load_model(model_name,device, **kwargs):
    if model_name == 'sd1_5x_2':
        model_class = StableDiffusionPipelineX_2
        model_id = "runwayml/stable-diffusion-v1-5"
        model = model_class.from_pretrained(pretrained_model_name_or_path=model_id, torch_dtype = torch.float16,**kwargs)
        model.to(device)
        model.attn_fetch_x = AttnFetchSDX_3()
        config = model.text_encoder.config

        for i in range(config.num_hidden_layers):
            spda_x=CLIPSdpaAttentionX(config).to(model.text_encoder.device, dtype=next(model.text_encoder.parameters()).dtype)
            new_state_dict = {}
            for key,value in spda_x.state_dict().items():
                new_state_dict[key] = model.text_encoder.text_model.encoder.layers[i].self_attn.state_dict()[key]
            spda_x.load_state_dict(new_state_dict)
            model.text_encoder.text_model.encoder.layers[i].self_attn = spda_x
            
    else:
        raise RuntimeError('model not accepted')
    
        
    return model



def save_image(image,directory,file_name):

    os.makedirs(directory, exist_ok = True)
    file_path = os.path.join(directory,file_name)
    image.save(f'{file_path}.jpg')
    return image



def save_text_sa_avg(text_sa,
                 directory,
                 file_name,
                 eos_idx=None):

    os.makedirs(directory,exist_ok=True)

    text_sa = text_sa.detach().cpu().numpy()
    text_sa = text_sa[1:eos_idx,1:eos_idx]*255*2
    im = Image.fromarray(text_sa.astype(np.uint16))
    im = im.resize((256,256))
    if im.mode == 'I;16':
        im = im.convert('L')
    save_image(image=im,directory=directory, file_name = file_name)


def get_prompt_list_by_line(directory:str, file_name = None):
    if file_name is None:
        raise ValueError("Missing prompts file name")
    else:
        file_path = os.path.join(directory,file_name)+'.txt'
        with open(file_path,'r') as f:
            lines = f.read().splitlines()
     
        return lines
    
def load_json(directory:str, file_name:str):
    file_path = os.path.join(directory,file_name)+'.json'
    if not os.path.exists(file_path):
        FileNotFoundError(f"{file_path} does not exist")
    else:
        with open(file_path, 'rb') as f:
            data = json.load(f)
        return data





def get_token_ids(prompt,tokenizer,max_length = 120):
    """
    outputs a tensor of 1*len_prompt of token_ids.
    """
    token_ids = tokenizer(text = prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",)['input_ids']
    
    if isinstance(tokenizer,CLIPTokenizer):
        eos_id = 49407
    else:
        eos_id = 1

    eos_idx = torch.nonzero(token_ids[0]==eos_id)[0]
        
    # breakpoint()
    token_ids = token_ids[:,:eos_idx+1] # tensor shape 1, eos_idx
    
    return token_ids, eos_idx


    
    

