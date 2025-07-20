

import torch
from typing import Dict

from models.processors import AttnProcessorX_2

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

        
        
class AttnFetchPixartX():
    def __init__(self,positive_prompt:bool = True):
        self.storage = {}
        self.positive_prompt = positive_prompt
        
    def get_timestep(self):
        timestep = self.unet.timestep.item()
        
        return timestep
    
    def maps_by_block(self):
        """
        final output. organizes data in dicts which keys are blocks and values are numpy arrays of maps in all times
        """
        data = rearrange_by_layers(self.storage)
        data = to_cpu_numpy(data)

        return data
     
    def store_attn_by_timestep(self,time,transformer):
        """
        saves data from get_unet_data_x fro all timesteps in a dict where keys are timesteps.
        """
        attn_data = self.get_attn_data(transformer)
        self.storage[time] = attn_data

    def get_attn_data(self,transformer):
        attn_data = {}
        
        for i, block in enumerate(transformer.transformer_blocks):
                data = block.attn2.processor.attn_data_x
                attn_data[f'block_{i}'] = data
        
        return attn_data

    def set_processor(self,transformer,processor_name, index_data):
        processor_classes = {'processor_x':AttnProcessorX_2,
                                 'processor_3': AttnProcessor3,}
       
        processors = {}

        for layer in transformer.attn_processors.keys():
            processor = processor_classes[processor_name](idx1 =index_data['obj_1'], idx2 = index_data['obj_2'],eos_idx = index_data['eos'])

            processors[layer] = processor
            
        transformer.set_attn_processor(processors)