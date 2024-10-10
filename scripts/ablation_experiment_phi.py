# Imports
from torch.utils.data import Dataset
from typing import Callable, Dict, List, Tuple
import torch
from pycocotools.coco import COCO 
import json
from PIL import Image
from tqdm import tqdm
import itertools
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HookedLVLM import HookedLVLM
from InputsEmbeds import InputsEmbeds
from ImageDatasets import COCOImageDataset
from utils import correct_annotations_for_crop, find_overlapping_patches, get_register_indices, get_object_patch_indices, replace_image_regions_with_patches, get_random_indices

from transformers import PreTrainedTokenizerBase
import numpy as np

from pycocotools import mask as maskUtils
import time
import yaml

def integrated_gradients(model, image, prompt, image_mean_tensor, model_id, steps=50):
    # Get the input embeddings for the actual input
    with torch.no_grad():
        input_embeds = model.get_text_model_in(image, prompt)
        _, _, img_start_end_indices = InputsEmbeds(model.processor.tokenizer, input_embeds, prompt, model_id).get_img_and_text_embed()
        img_start, img_end = img_start_end_indices[0]
        baseline_embeds = input_embeds.clone()

        replacement_tensor = image_mean_tensor.to(baseline_embeds.dtype).to(baseline_embeds.device)
        replacement_tensor.unsqueeze(0).expand((img_end-img_start+1), -1)

        baseline_embeds[:, img_start:img_end+1, :] = replacement_tensor

    # Compute the difference between input and baseline
    diff = input_embeds - baseline_embeds
    
    # Initialize the integrated gradients
    integrated_grads = torch.zeros_like(input_embeds)
    
    # Iterate through steps
    for alpha in torch.linspace(0, 1, steps):
        # Compute the interpolated input
        interpolated_embeds = baseline_embeds + alpha * diff
        interpolated_embeds.requires_grad_(True)
        
        # Forward pass
        outputs = model.model(inputs_embeds=interpolated_embeds)
        logits = outputs.logits
        
        # Assuming the model outputs logits for Yes/No, we'll focus on the "Yes" logit
        yes_logit = logits[0, -1, model.processor.tokenizer.encode("Yes")[0]]
        
        # Compute gradients
        model.model.zero_grad()
        yes_logit.backward(retain_graph=True)
        
        # Accumulate the gradients
        integrated_grads += interpolated_embeds.grad
    
    # Compute the average gradients
    integrated_grads *= diff / steps
    
    # Don't have to correct for image indices since we only did gradient inputs for image indices already
    return integrated_grads

def get_high_gradient_indices(model, image, ann, class_name, image_mean_tensor, model_id, steps=50):
    
    # Prepare the input
    text_question = f"Is there a {class_name} in this image?"
    prompt = f"<|user|>\n<image>{text_question}<|end|>\n<|assistant|>\n"
    
    # Compute integrated gradients
    integrated_grads = integrated_gradients(model, image, prompt, image_mean_tensor, model_id, steps)
    
    # Sum the gradients across the embedding dimension
    token_importance = integrated_grads.abs().sum(dim=-1).squeeze()
    
    # Get the indices sorted by importance (highest to lowest)
    sorted_indices = torch.argsort(token_importance, descending=True)
    
    # Convert to list and return
    return sorted_indices.tolist()

def check_identification(model, image, class_name, ablate_indices, replacement_tensor):

    text_question_1 = "Describe this image."
    prompt_1 = f"<|user|>\n<image>{text_question_1}<|end|>\n<|assistant|>\n"

    if ablate_indices is None or ablate_indices == []:
        answer_1 = model.generate(image, prompt_1, max_new_tokens=200, do_sample=False)
    else:
        with model.ablate_inputs(indices=ablate_indices, replacement_tensor=replacement_tensor):
            answer_1 = model.generate(image, prompt_1, max_new_tokens=200, do_sample=False)

    check_1 = class_name in answer_1

    text_question_2 = f"Is there a {class_name} in this image?"
    prompt_2 = f"<|user|>\n<image>{text_question_2}<|end|>\n<|assistant|>\n"

    if ablate_indices is None or ablate_indices == []:
        answer_2 = model.generate(image, prompt_2, max_new_tokens=10, do_sample=False)
    else:
        with model.ablate_inputs(indices=ablate_indices, replacement_tensor=replacement_tensor):
            answer_2 = model.generate(image, prompt_2, max_new_tokens=10, do_sample=False)

    check_2 = "yes" in answer_2.lower()

    return {"generative": check_1, "polling": check_2, "description": answer_1}

# Main!
import argparse
import json
import os
import itertools
from tqdm import tqdm
import torch

def main(device, data_dir, data_type, results_file, mean_tensor_loc, zero_ablation):
    # Load and Filter COCO Dataset
    ann_file = '{}/annotations/instances_{}.json'.format(data_dir, data_type)

    cfg = {
        "zero_ablation": zero_ablation,
    }

    def filter_fn(anns, return_ann=False):
        # Filters for area, objects with only one instance, and images with less than 4 annotations.
        area_threshold = (1000, 2000)

        category_groups = {}

        # Filter out those with more than 3 annotations
        if len(anns) > 3:
            return False

        for ann in anns:
            cat_id = ann['category_id']
            if cat_id not in category_groups:
                category_groups[cat_id] = []
            category_groups[cat_id].append(ann)
        
        # Choose for images that has an object with only one instance and area within threshold
        for cat_id, group in category_groups.items():
            if len(group) == 1 and (area_threshold[0] < group[0]['area'] < area_threshold[1]):
                if return_ann:
                    return group[0]
                else:
                    return True

        return False

    ds = COCOImageDataset(data_dir=data_dir, data_type=data_type, ann_file=ann_file, filter_fn=filter_fn)

    # Load Model
    model_id = "xtuner/llava-phi-3-mini-hf"
    model = HookedLVLM(model_id=model_id, device=device, quantize=True, quantize_type="fp16")
    model.model.eval()
    image_mean_tensor = torch.load(mean_tensor_loc)

    # Load existing results if available
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
        if "cfg" in results:
            zero_ablation = results["cfg"]["zero_ablation"]
    else:
        results = {"cfg": cfg}

    if zero_ablation:
        image_mean_tensor = torch.zeros_like(image_mean_tensor)
        print("Zero ablation enabled. Using zero tensor for ablation.")

    success_count = 0
    for img_id, image, annotations in tqdm(ds):
        # Skip if we reach 1,000 successful images
        if success_count >= 1000:
            break

        # Skip if this image has already been processed
        if str(img_id) in results:
            continue

        ann = filter_fn(annotations, return_ann=True)
        category_id = ann['category_id']
        class_name = ds.coco.loadCats(category_id)[0]['name']

        img_results = {}

        corrected_img, corrected_ann = correct_annotations_for_crop(image, ann)
        if len(corrected_ann) == 0:
            continue # Skip if no valid annotation found

        # Establish baseline
        _, overlapping_patch_indices = find_overlapping_patches(corrected_img, corrected_ann[0])
        ablated_image = replace_image_regions_with_patches(corrected_img, overlapping_patch_indices, patch_type="noise")

        start_time = time.time()
        with torch.no_grad():
            no_ablation_result = check_identification(model, image, class_name, None, None)
            if no_ablation_result['generative'] is False: # if model doesn't identify the object, skip
                continue

            img_ablation_result = check_identification(model, ablated_image, class_name, None, None)
            if img_ablation_result['generative'] is True: # if model still identifies the object, skip
                continue

        check_time = time.time() - start_time
        print(f"Image {img_id} took {check_time:.2f} seconds to check.")

        with torch.no_grad():
            text_question = "Describe this image."
            prompt = f"<|user|>\n<image>{text_question}<|end|>\n<|assistant|>\n"
            activations = model.get_text_model_in(image, prompt)
            inputs_embeds = InputsEmbeds(model.processor.tokenizer, activations, prompt, model_id)
            img_embeds, text_embeds, start_end_indices = inputs_embeds.get_img_and_text_embed()

            register_ablation_indices = get_register_indices(img_embeds, text_embeds, start_end_indices)
            patch_ablation_indices_0 = get_object_patch_indices(image, ann, start_end_indices, buffer=0)
            patch_ablation_indices_1 = get_object_patch_indices(image, ann, start_end_indices, buffer=1)
            patch_ablation_indices_2 = get_object_patch_indices(image, ann, start_end_indices, buffer=2)
        
        gradient_indices = get_high_gradient_indices(model, image, ann, class_name, image_mean_tensor, model_id)

        gradient_ablation_indices_5 = gradient_indices[:5]
        gradient_ablation_indices_10 = gradient_indices[:10]
        gradient_ablation_indices_20 = gradient_indices[:20]
        gradient_ablation_indices_40 = gradient_indices[:40]
        gradient_ablation_indices_60 = gradient_indices[:60]
        gradient_ablation_indices_100 = gradient_indices[:100]
        gradient_ablation_indices_250 = gradient_indices[:250]

        random_indices_5 = get_random_indices(start_end_indices, 5)
        random_indices_10 = get_random_indices(start_end_indices, 10)
        random_indices_20 = get_random_indices(start_end_indices, 20)
        random_indices_40 = get_random_indices(start_end_indices, 40)
        random_indices_60 = get_random_indices(start_end_indices, 60)
        random_indices_100 = get_random_indices(start_end_indices, 100)
        random_indices_250 = get_random_indices(start_end_indices, 250)
       
        with torch.no_grad():
            img_results["no_ablation"] = no_ablation_result
            img_results["image_ablation"] = img_ablation_result
            img_results["register_ablation"] = check_identification(model, image, class_name, register_ablation_indices, image_mean_tensor)
            img_results["patch_ablation_0"] = check_identification(model, image, class_name, patch_ablation_indices_0, image_mean_tensor)
            img_results["patch_ablation_1"] = check_identification(model, image, class_name, patch_ablation_indices_1, image_mean_tensor)
            img_results["patch_ablation_2"] = check_identification(model, image, class_name, patch_ablation_indices_2, image_mean_tensor)
            img_results["gradient_ablation_5"] = check_identification(model, image, class_name, gradient_ablation_indices_5, image_mean_tensor)
            img_results["gradient_ablation_10"] = check_identification(model, image, class_name, gradient_ablation_indices_10, image_mean_tensor)
            img_results["gradient_ablation_20"] = check_identification(model, image, class_name, gradient_ablation_indices_20, image_mean_tensor)       
            img_results["gradient_ablation_40"] = check_identification(model, image, class_name, gradient_ablation_indices_40, image_mean_tensor)       
            img_results["gradient_ablation_60"] = check_identification(model, image, class_name, gradient_ablation_indices_60, image_mean_tensor)       
            img_results["gradient_ablation_100"] = check_identification(model, image, class_name, gradient_ablation_indices_100, image_mean_tensor)       
            img_results["gradient_ablation_250"] = check_identification(model, image, class_name, gradient_ablation_indices_250, image_mean_tensor)       
            img_results["random_ablation_5"] = check_identification(model, image, class_name, random_indices_5, image_mean_tensor)
            img_results["random_ablation_10"] = check_identification(model, image, class_name, random_indices_10, image_mean_tensor)
            img_results["random_ablation_20"] = check_identification(model, image, class_name, random_indices_20, image_mean_tensor)
            img_results["random_ablation_40"] = check_identification(model, image, class_name, random_indices_40, image_mean_tensor)
            img_results["random_ablation_60"] = check_identification(model, image, class_name, random_indices_60, image_mean_tensor)
            img_results["random_ablation_100"] = check_identification(model, image, class_name, random_indices_100, image_mean_tensor)
            img_results["random_ablation_250"] = check_identification(model, image, class_name, random_indices_250, image_mean_tensor)

        # Save the indices too, all in one dictionary
        img_results["indices"] = {
            "register_ablation": register_ablation_indices,
            "patch_ablation_0": patch_ablation_indices_0,
            "patch_ablation_1": patch_ablation_indices_1,
            "patch_ablation_2": patch_ablation_indices_2,
            "gradient_ablation_5": gradient_ablation_indices_5,
            "gradient_ablation_10": gradient_ablation_indices_10,
            "gradient_ablation_20": gradient_ablation_indices_20,
            "gradient_ablation_40": gradient_ablation_indices_40,
            "gradient_ablation_60": gradient_ablation_indices_60,
            "gradient_ablation_100": gradient_ablation_indices_100,
            "gradient_ablation_250": gradient_ablation_indices_250,
            "random_ablation_5": random_indices_5,
            "random_ablation_10": random_indices_10,
            "random_ablation_20": random_indices_20,
            "random_ablation_40": random_indices_40,
            "random_ablation_60": random_indices_60,
            "random_ablation_100": random_indices_100,
            "random_ablation_250": random_indices_250,
        }

        results[img_id] = img_results

        success_count += 1

        end_time = time.time() - start_time
        print(f"Full Run: Image {img_id} took {end_time:.2f} seconds to process.")

        # Save results after each iteration
        with open(results_file, "w") as f:
            json.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process images and annotations.")
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use (e.g., cuda:0, cpu)')
    parser.add_argument('--datatype', type=str, default='val2017', help='Data type for COCO dataset')
    parser.add_argument('--results_file', type=str, default="results_val.json", help='File to save results')
    parser.add_argument('--mean_tensor', type=str, help='Filepath of the Mean Tensor for ablation')
    parser.add_argument('--zero_ablation', action='store_true', help='Flag to enable zero ablation')

    args = parser.parse_args()

    config_file = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    data_dir = config['data_dir']

    main(args.device, data_dir, args.datatype, args.results_file, args.mean_tensor, args.zero_ablation)