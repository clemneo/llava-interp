import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from IPython.display import display
import torch
from PIL import Image
from contextlib import contextmanager
from typing import Callable, Union, Dict, Any
import os
import yaml
file_path = os.path.dirname(__file__)
config_file = os.path.join(file_path, 'config.yaml')
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

model_cache_dir = config['cache_dir']
if model_cache_dir is None:
    model_cache_dir = os.path.join(file_path, '..', 'models')


@contextmanager
def session_hook(model: torch.nn.Module, hook: Callable):
    handle = model.register_forward_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()

class BlockAttentionHook:
    def __init__(self, indices_list):
        self.indices_list = indices_list # [(i, j), ]

    def __call__(self, module, args, kwargs):
        hidden_states = kwargs['hidden_states']
        bsz, seq_len, _ = hidden_states.shape
        attention_mask = kwargs.get("attention_mask")
        
        if attention_mask is None:
            attention_mask = torch.ones(bsz, 1, seq_len, seq_len, dtype=torch.bool, device=hidden_states.device).tril(diagonal=0)
        else:
            attention_mask = attention_mask.clone()

        for i, j in self.indices_list:
            # print("Setting attention mask from token {} to token {} to False".format(i, j))
            attention_mask[:, :, i, j] = False

        kwargs["attention_mask"] = attention_mask
        return args, kwargs

class HookedLVLM:
    """Hooked LVLM.
    
    
    """
    def __init__(self, 
                 model_id: str = "llava-hf/llava-1.5-7b-hf",
                 hook_loc: str = "text_model_in",
                 device: str = "cuda:0",
                 quantize: bool = False,
                 quantize_type: str = "fp16",
                 ):
        if quantize:
            if quantize_type == "4bit":
                # Initialize the model and processor
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16, 
                    low_cpu_mem_usage=True,
                    quantization_config=bnb_config,
                    device_map=device,
                    cache_dir=model_cache_dir
                )
            elif quantize_type == "fp16":
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16, 
                    low_cpu_mem_usage=True,
                    device_map=device,
                    cache_dir=model_cache_dir
                )
            elif quantize_type == "int8":
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.int8, 
                    low_cpu_mem_usage=True,
                    device_map=device,
                    cache_dir=model_cache_dir
                )
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id, 
                device_map=device,
                cache_dir=model_cache_dir)

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.hook_loc = hook_loc 
        self.data = None

    @contextmanager
    def ablate_inputs(self, indices, replacement_tensor):
        def ablation_hook(module, args, kwargs):
            input_embeds = kwargs["inputs_embeds"]
            if input_embeds.shape[-2] == 1: # If this is not the first forward pass
                return args, kwargs
            modified_input = input_embeds.clone()
            # Convert replacement tensor to the same dtype and device as the input tensor
            local_replacement_tensor = replacement_tensor.to(modified_input.dtype).to(modified_input.device)
            local_replacement_tensor.unsqueeze(0).expand(len(indices), -1)
            modified_input[:, indices, :] = local_replacement_tensor
            kwargs["inputs_embeds"] = modified_input
            return args, kwargs

        # Register the forward pre-hook
        hook = self.model.language_model.register_forward_pre_hook(ablation_hook, with_kwargs=True)
        try:
            yield
        finally:
            # Remove the hook when exiting the context
            hook.remove()

    @contextmanager
    def block_attention(self, attn_block_dict):
        """
        attn_block_dict: {layer: [(i, j), ...]} where setting (i, j) 
        stops information from token j from going to token i, or
        (equivalently) stops token j from attending to token i.
        Then i >= j for causal LMs.
        """
        hooks = []

        # Register hooks for each layer
        for layer, indices_list in attn_block_dict.items():
            hook = BlockAttentionHook(indices_list)
            h = self.model.language_model.model.layers[layer].self_attn.register_forward_pre_hook(
                hook, with_kwargs=True)
            hooks.append(h)

        has_error = False
        try:
            yield
        except Exception as e:
            import traceback
            traceback.print_exc()
            has_error = True
        finally:
            # Remove all hooks
            for h in hooks:
                h.remove()
            if has_error:
                raise Exception("An error occurred during the block_attention context")
        

    def prompt_hook(self, module, args, kwargs, output):
        self.data = kwargs['inputs_embeds']
        return output
    
    def forward(self, 
                image_path_or_image: Union[str, Image.Image], 
                prompt: str,
                output_hidden_states=False,
                output_attentions=False,
                ):
        
        # Open image if needed
        if isinstance(image_path_or_image, str):
            image = Image.open(image_path_or_image)
        else:
            image = image_path_or_image

        # Prepare inputs
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs.to(self.model.device)

        # Run forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=output_hidden_states, output_attentions=output_attentions)

        return outputs
        # if output_hidden_states:
        #     return outputs.hidden_states
        # if output_attentions:
        #     return outputs.attentions
        # else:
        #     return self.processor.batch_decode(outputs.logits[-1].argmax(dim=-1), skip_special_tokens=True, 
        #                                    clean_up_tokenization_spaces=False)[0]

    def generate(self, 
                 image_path_or_image: Union[str, Image.Image], 
                 prompt: str,
                 max_new_tokens: int = 100,
                 output_hidden_states: bool = False,
                 do_sample=True
                 ):
        
        # Open image if needed
        if isinstance(image_path_or_image, str):
            image = Image.open(image_path_or_image)
        else:
            image = image_path_or_image

        # Prepare inputs
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs.to(self.model.device)

        # Run forward pass
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, output_hidden_states=output_hidden_states, return_dict_in_generate=True, do_sample=do_sample)

        response_str = self.processor.batch_decode(output.sequences, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)[0]
        
        if output_hidden_states:
            return response_str, output.hidden_states
        
        return response_str
        
    
    def get_text_model_in(self, 
                          image_path_or_image: 
                          Union[str, Image.Image], 
                          prompt: str,
                          ):
        
        # Open image if needed
        if isinstance(image_path_or_image, str):
            image = Image.open(image_path_or_image)
        else:
            image = image_path_or_image

        # Prepare inputs
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs.to(self.model.device)
        
        # Run forward pass with hook
        if self.hook_loc == "text_model_in":
            with session_hook(self.model.language_model, self.prompt_hook):
                with torch.no_grad():
                    outputs = self.model(**inputs)
        else:
            raise ValueError(f"Only 'text_model_in' support for hook location at the moment. \
                             Got {self.hook_loc} instead.")
        

        return self.data
    
    # actually unused atm, I think. Not updated for phi-llava yet
    def get_image_tokens(self, batch_images):
        """Takes in a list of Images, outputs their image tokens with shape (num_images, num_tokens, hidden_size)"""
        inputs = self.processor(images=batch_images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)

        a_out = self.model.vision_tower(pixel_values, output_hidden_states=True)
        a_out = a_out.hidden_states[-2][:, 1:] # The second last layer is what LLAVA actually uses. Removing the first token as it is [CLS].
        return self.model.multi_modal_projector(a_out)
   

# %%
