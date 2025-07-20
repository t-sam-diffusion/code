

import torch
from typing import Dict

from models.processors import AttnProcessorX_2




PROCESSOR_CLASSES = {'processor_x_2': AttnProcessorX_2}


def to_cpu_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, dict):
        return {key: to_cpu_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_cpu_numpy(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_cpu_numpy(item) for item in data)
    else:
        return data
    
    
def rearrange_by_layers(data:dict) -> Dict:
    layers = list(list(data.values())[0].keys())
    rearranged_output = {}
    for layer in layers:
        attn_tensors = torch.stack([data[time][layer] for time in data.keys()])
        rearranged_output[layer] = attn_tensors


    return rearranged_output


    
class AttnFetchSDX_3():
    def __init__(self,positive_prompt:bool = True):
        self.storage = {}
        self.positive_prompt = positive_prompt


    def maps_by_block(self):
        """
        final output. organizes data in dicts which keys are blocks and values are numpy arrays of maps in all times
        """

        data = rearrange_by_layers(self.storage)
        data = to_cpu_numpy(data)
        
        return data
        
    def store_attn_by_timestep(self,time,unet):
        """
        saves data from get_unet_data_x fro all timesteps in a dict where keys are timesteps.
        """
        attn_data = self.get_unet_data(unet)
        # attn_data = self.get_unet_data_2(unet)
        self.storage = attn_data ## due to memory issue
    
    def store_key_by_timestep(self,time,unet):
        """
        saves data from get_unet_data_x fro all timesteps in a dict where keys are timesteps.
        """
        key = self.get_unet_data_key(unet)
        self.storage_key = key ## due to memory issue    


    def get_unet_data(self,unet):
        """
        saves attention maps in a dict where the keys are unet blocks: down_0 .. down_2, mid, up_1 ... up_3.
        values are attention maps for either positive or negative prompt, averaged over heads, at the current timestep.
        """
        unet_attn_data = []
        for _, block in enumerate(unet.down_blocks):
            if block.__class__.__name__ == "CrossAttnDownBlock2D":
                if block.attentions[0].transformer_blocks[0].attn2.processor.attn_data_x.size()[0] == 256:
                    for i in range(2):
                        data = block.attentions[i].transformer_blocks[0].attn2.processor.attn_data_x
                        unet_attn_data.append(data.unsqueeze(0))
                    
                    
        for _, block in enumerate(unet.up_blocks):
            if block.__class__.__name__ == "CrossAttnUpBlock2D":
                if block.attentions[0].transformer_blocks[0].attn2.processor.attn_data_x.size()[0] == 256:
                    for i in range(3):
                        data = block.attentions[i].transformer_blocks[0].attn2.processor.attn_data_x
                        unet_attn_data.append(data.unsqueeze(0))

        unet_attn_data = torch.cat(unet_attn_data, dim=0)
        unet_attn_data = unet_attn_data.sum(0) / unet_attn_data.shape[0]
        unet_attn_data = unet_attn_data.reshape(16,16,-1)
        return unet_attn_data
    
    
    def get_unet_data_2(self,unet):
        """
        saves attention maps in a dict where the keys are unet blocks: down_0 .. down_2, mid, up_1 ... up_3.
        values are attention maps for either positive or negative prompt, averaged over heads, at the current timestep.
        """
        unet_attn_data = []
        # for _, block in enumerate(unet.down_blocks):
        #     if block.__class__.__name__ == "CrossAttnDownBlock2D":
        #         if block.attentions[0].transformer_blocks[0].attn2.processor.attn_data_x.size()[0] == 256:
        #             for i in range(2):
        #                 data = block.attentions[i].transformer_blocks[0].attn2.processor.attn_data_x
        #                 unet_attn_data.append(data.unsqueeze(0))
                    
                    
        for _, block in enumerate(unet.up_blocks):

            if block.__class__.__name__ == "CrossAttnUpBlock2D":
                if block.attentions[0].transformer_blocks[0].attn2.processor.attn_data_x.size()[0] == 256:
                    # for i in range(3):
                    data = block.attentions[0].transformer_blocks[0].attn2.processor.attn_data_x
                    unet_attn_data.append(data.unsqueeze(0))
        # breakpoint()
        unet_attn_data = torch.cat(unet_attn_data, dim=0)
        unet_attn_data = unet_attn_data.sum(0) / unet_attn_data.shape[0]
        unet_attn_data = unet_attn_data.reshape(16,16,-1)
        return unet_attn_data
    
    def get_unet_data_key(self,unet):
        """
        saves attention maps in a dict where the keys are unet blocks: down_0 .. down_2, mid, up_1 ... up_3.
        values are attention maps for either positive or negative prompt, averaged over heads, at the current timestep.
        """
        unet_attn_data = {}
        for i, block in enumerate(unet.down_blocks):
            if block.__class__.__name__ == "CrossAttnDownBlock2D":
                # if block.attentions[0].transformer_blocks[0].attn2.processor.attn_data_x.size()[0] == 256:
                data = block.attentions[0].transformer_blocks[0].attn2.processor.key_store
                data += block.attentions[1].transformer_blocks[0].attn2.processor.key_store
                    
                unet_attn_data[f'down_{i}'] = data/2
                    
        block = unet.mid_block
        data = block.attentions[0].transformer_blocks[0].attn2.processor.key_store
        unet_attn_data[f'mid'] = data
        for i, block in enumerate(unet.up_blocks):
            if block.__class__.__name__ == "CrossAttnUpBlock2D":
                data = block.attentions[0].transformer_blocks[0].attn2.processor.key_store
                data += block.attentions[1].transformer_blocks[0].attn2.processor.key_store
                data += block.attentions[2].transformer_blocks[0].attn2.processor.key_store
                                   
                unet_attn_data[f'up_{i}'] = data/3 
                 
        import torch.nn.functional as F

        for key, value in unet_attn_data.items():   
            print(key, F.cosine_similarity(value[0][2], value[0][5], dim=0))
        exit()
        return unet_attn_data
    
    
    def set_processor(self,unet,processor_name = 'processor_x_2'):
        
        processors= {}
        for layer in unet.attn_processors.keys():
            processor = PROCESSOR_CLASSES[processor_name]()
            processors[layer] = processor
        unet.set_attn_processor(processors)
        
    def store_text_sa(self,text_encoder):
        attn_data = {}
    
        for i, block in enumerate(text_encoder.text_model.encoder.layers):
            data = block.self_attn.attn_data_x
            attn_data[f'block_{i}'] = data

        return attn_data