import argparse
import json
import os
import torch
from tqdm import tqdm
from torch.nn import functional as F
from PIL import Image

from typing import Callable, Dict, List, Tuple
import itertools
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from HookedLVLM import HookedLVLM
from ImageDatasets import COCOImageDataset
from utils import correct_annotations_for_crop, get_object_patch_indices
from InputsEmbeds import InputsEmbeds


def format_and_compare_answers(model, prompt, answer, class_name):
    generated_answer = answer.strip().lower()

    # Tokenize the class name and get the first token
    tokens = model.processor.tokenizer.tokenize(class_name.lower())
    first_token = tokens[0].strip()

    # Remove leading non-alphabetic characters
    first_token = first_token.lstrip('Ġ▁')

    # Check if the answer starts with the class name token
    result = first_token.lower().startswith(generated_answer.lower())

    return result

def get_attention_block_dict(from_indices: List[int], to_indices: List[int], layer_list: List[int]) -> Dict:
    block_list = list(itertools.product(to_indices, from_indices))
    return {layer: block_list for layer in layer_list}


def run_experiment(model, image, prompt, class_name, attn_block_dict=None):
    with torch.no_grad():
        if attn_block_dict:
            with model.block_attention(attn_block_dict):
                y = model.forward(image, prompt)
        else:
            y = model.forward(image, prompt)

        logits = y['logits']
        generated_token = model.processor.batch_decode(logits[:, -1, :].argmax(dim=-1), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        probs = F.softmax(logits[:, -1, :], dim=-1)
        generated_token_prob = probs[:, logits[:, -1, :].argmax(dim=-1)].item()

        first_token = model.processor.tokenizer.tokenize(class_name.lower())[0].strip()
        correct_token_id = model.processor.tokenizer.convert_tokens_to_ids(first_token)
        correct_token_prob = probs[0, correct_token_id].item()

        is_correct = format_and_compare_answers(model, prompt, generated_token, class_name)

        return {
            "is_correct": is_correct,
            "generated_token": generated_token,
            "generated_token_prob": generated_token_prob,
            "correct_token": first_token,
            "correct_token_prob": correct_token_prob
        }

def main(device, datatype, results_file, clean_questions_file):
    # Load COCO Dataset
    dataDir = '/scratch/local/ssd2/clement'
    dataType = datatype
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

    def filter_fn(anns, return_ann=False):
        area_threshold = (1000, 2000)
        category_groups = {}
        if len(anns) > 3:
            return False
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id not in category_groups:
                category_groups[cat_id] = []
            category_groups[cat_id].append(ann)
        for cat_id, group in category_groups.items():
            if len(group) == 1 and (area_threshold[0] < group[0]['area'] < area_threshold[1]):
                return group[0] if return_ann else True
        return False

    ds = COCOImageDataset(data_dir=dataDir, data_type=dataType, ann_file=annFile, filter_fn=filter_fn)

    # Load Model
    model = HookedLVLM(device=device, quantize=True, quantize_type="fp16")
    model.model.eval()

    # Load clean questions
    with open(clean_questions_file, 'r') as f:
        clean_questions = json.load(f)

    # Load existing results if available
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        results = {}

    experiment_keys = set()

    # Process each image
    for img_id, image, annotations in tqdm(ds):
        if str(img_id) not in clean_questions or len(clean_questions[str(img_id)]) < 1:
            continue

        ann = filter_fn(annotations, return_ann=True)
        if ann is None:
            continue

        category_id = ann['category_id']
        class_name = ds.coco.loadCats(category_id)[0]['name'].lower()

        corrected_img, corrected_ann = correct_annotations_for_crop(image, ann)
        if len(corrected_ann) == 0:
            continue

        # Initialize results for this image if not present
        if str(img_id) not in results:
            results[str(img_id)] = {}

        text_question = clean_questions[str(img_id)][0]
        prompt = f"USER: <image>\n{text_question} ASSISTANT: It is a"

        # Run baseline experiment
        baseline_result = run_experiment(model, image, prompt, class_name)
        results[str(img_id)]["baseline"] = baseline_result

        # Skip attention experiments if baseline is incorrect
        if not baseline_result["is_correct"]:
            continue

        # Prepare attention blocking experiments
        activations = model.get_text_model_in(image, prompt)
        input_embeds = InputsEmbeds(model.processor.tokenizer, activations, prompt)
        _, _, start_end_indices = input_embeds.get_img_and_text_embed()
        num_tokens = activations.shape[1]
        num_layers = 32

        # Define presets
        patch_indices = get_object_patch_indices(image, ann, start_end_indices, buffer=0)
        patch_plus_one_indices = get_object_patch_indices(image, ann, start_end_indices, buffer=1)
        patch_plus_two_indices = get_object_patch_indices(image, ann, start_end_indices, buffer=2)

        start, end = start_end_indices[0]
        all_minus_patch_plus_one = list(set(range(start, end)) - set(patch_plus_one_indices))

        early_layers = list(range(10))
        mid_layers = list(range(11, 21))
        late_layers = list(range(21, 32))
        all_layers = list(range(num_layers))
        # do 5-15, 15-25
        early_mid_layers = list(range(5, 15))
        mid_late_layers = list(range(15, 25))

        experiments = {
            "PATCH_PLUS_ONE_TO_LAST_ROW_EARLY_MID_LAYERS": {
                "from": get_object_patch_indices(image, ann, start_end_indices, buffer=1),
                "to": list(range(start_end_indices[0][1]-24, start_end_indices[0][1])),
                "layers": early_mid_layers
            },
            "PATCH_PLUS_ONE_TO_LAST_ROW_MID_LATE_LAYERS": {
                "from": get_object_patch_indices(image, ann, start_end_indices, buffer=1),
                "to": list(range(start_end_indices[0][1]-24, start_end_indices[0][1])),
                "layers": mid_late_layers
            },
            "ALL_MINUS_PATCH_EARLY_LAYERS": {
                "from": all_minus_patch_plus_one,
                "to": [num_tokens-1],
                "layers": early_layers,
            },
            "ALL_MINUS_PATCH_EARLY_MID_LAYERS": {
                "from": all_minus_patch_plus_one,
                "to": [num_tokens-1],
                "layers": early_mid_layers,
            },
            "ALL_MINUS_PATCH_MID_LAYERS": {
                "from": all_minus_patch_plus_one,
                "to": [num_tokens-1],
                "layers": mid_layers,
            },
            "ALL_MINUS_PATCH_MID_LATE_LAYERS": {
                "from": all_minus_patch_plus_one,
                "to": [num_tokens-1],
                "layers": mid_late_layers,
            },
            "ALL_MINUS_PATCH_LATE_LAYERS": {
                "from": all_minus_patch_plus_one,
                "to": [num_tokens-1],
                "layers": late_layers,
            },
            "ALL_MINUS_PATCH_ALL_LAYERS": {
                "from": all_minus_patch_plus_one,
                "to": [num_tokens-1],
                "layers": all_layers,
            },
            "PATCH_TO_LAST_TOK_EARLY_MID_LAYERS": {
                "from": patch_indices,
                "to": [num_tokens - 1],
                "layers": early_mid_layers
            },
            "PATCH_TO_LAST_TOK_MID_LATE_LAYERS": {
                "from": patch_indices,
                "to": [num_tokens - 1],
                "layers": mid_late_layers
            },
            "PATCH_PLUS_ONE_TO_LAST_TOK_EARLY_MID_LAYERS": {
                "from": patch_plus_one_indices,
                "to": [num_tokens - 1],
                "layers": early_mid_layers
            },
           "PATCH_PLUS_ONE_TO_LAST_TOK_MID_LATE_LAYERS": {
                "from": patch_plus_one_indices,
                "to": [num_tokens - 1],
                "layers": mid_late_layers
            },
            "PATCH_PLUS_TWO_TO_LAST_TOK_EARLY_MID_LAYERS": {
                "from": patch_plus_two_indices,
                "to": [num_tokens - 1],
                "layers": early_mid_layers
            },
            "PATCH_PLUS_TWO_TO_LAST_TOK_MID_LATE_LAYERS": {
                "from": patch_plus_two_indices,
                "to": [num_tokens - 1],
                "layers": mid_late_layers
            },
            "PATCH_TO_LAST_TOK_ALL_LAYERS": {
                "from": patch_indices,
                "to": [num_tokens - 1],
                "layers": all_layers
            },
            "PATCH_TO_LAST_TOK_EARLY_LAYERS": {
                "from": patch_indices,
                "to": [num_tokens - 1],
                "layers": early_layers
            },
            "PATCH_TO_LAST_TOK_MID_LAYERS": {
                "from": patch_indices,
                "to": [num_tokens - 1],
                "layers": mid_layers
            },
            "PATCH_TO_LAST_TOK_LATE_LAYERS": {
                "from": patch_indices,
                "to": [num_tokens - 1],
                "layers": late_layers
            },
            "PATCH_PLUS_ONE_TO_LAST_TOK_ALL_LAYERS": {
                "from": patch_plus_one_indices,
                "to": [num_tokens - 1],
                "layers": all_layers
            },
            "PATCH_PLUS_ONE_TO_LAST_TOK_EARLY_LAYERS": {
                "from": patch_plus_one_indices,
                "to": [num_tokens - 1],
                "layers": early_layers
            },
            "PATCH_PLUS_ONE_TO_LAST_TOK_MID_LAYERS": {
                "from": patch_plus_one_indices,
                "to": [num_tokens - 1],
                "layers": mid_layers
            },
            "PATCH_PLUS_ONE_TO_LAST_TOK_LATE_LAYERS": {
                "from": patch_plus_one_indices,
                "to": [num_tokens - 1],
                "layers": late_layers
            },
            "PATCH_PLUS_TWO_TO_LAST_TOK_ALL_LAYERS": {
                "from": patch_plus_two_indices,
                "to": [num_tokens - 1],
                "layers": all_layers
            },
            "PATCH_PLUS_TWO_TO_LAST_TOK_EARLY_LAYERS": {
                "from": patch_plus_two_indices,
                "to": [num_tokens - 1],
                "layers": early_layers
            },
            "PATCH_PLUS_TWO_TO_LAST_TOK_MID_LAYERS": {
                "from": patch_plus_two_indices,
                "to": [num_tokens - 1],
                "layers": mid_layers
            },
            "PATCH_PLUS_TWO_TO_LAST_TOK_LATE_LAYERS": {
                "from": patch_plus_two_indices,
                "to": [num_tokens - 1],
                "layers": late_layers
            },
            "PATCH_PLUS_ONE_TO_LAST_ROW_ALL_LAYERS": { # For Basu
                "from": get_object_patch_indices(image, ann, start_end_indices, buffer=1),
                "to": list(range(start_end_indices[0][1]-24, start_end_indices[0][1])),
                "layers": list(range(num_layers))
            },
            "PATCH_PLUS_ONE_TO_LAST_ROW_EARLY_LAYERS": {
                "from": get_object_patch_indices(image, ann, start_end_indices, buffer=1),
                "to": list(range(start_end_indices[0][1]-24, start_end_indices[0][1])),
                "layers": early_layers
            },
            "PATCH_PLUS_ONE_TO_LAST_ROW_MID_LAYERS": {
                "from": get_object_patch_indices(image, ann, start_end_indices, buffer=1),
                "to": list(range(start_end_indices[0][1]-24, start_end_indices[0][1])),
                "layers": mid_layers
            },
            "PATCH_PLUS_ONE_TO_LAST_ROW_LATE_LAYERS": {
                "from": get_object_patch_indices(image, ann, start_end_indices, buffer=1),
                "to": list(range(start_end_indices[0][1]-24, start_end_indices[0][1])),
                "layers": late_layers
            },
            "PATCH_TO_LAST_ROW_ALL_LAYERS": {
                "from": list(range(start_end_indices[0][0], start_end_indices[0][1]-24)),
                "to": list(range(start_end_indices[0][1]-24, start_end_indices[0][1])),
                "layers": all_layers
            },   
            "PATCH_TO_LAST_ROW_EARLY_LAYERS": {
                "from": list(range(start_end_indices[0][0], start_end_indices[0][1]-24)),
                "to": list(range(start_end_indices[0][1]-24, start_end_indices[0][1])),
                "layers": early_layers
            },   
            "PATCH_TO_LAST_ROW_EARLY_MID_LAYERS": {
                "from": list(range(start_end_indices[0][0], start_end_indices[0][1]-24)),
                "to": list(range(start_end_indices[0][1]-24, start_end_indices[0][1])),
                "layers": early_mid_layers
            },   
            "PATCH_TO_LAST_ROW_MID_LAYERS": {
                "from": list(range(start_end_indices[0][0], start_end_indices[0][1]-24)),
                "to": list(range(start_end_indices[0][1]-24, start_end_indices[0][1])),
                "layers": mid_layers
            },   
            "PATCH_TO_LAST_ROW_MID_LATE_LAYERS": {
                "from": list(range(start_end_indices[0][0], start_end_indices[0][1]-24)),
                "to": list(range(start_end_indices[0][1]-24, start_end_indices[0][1])),
                "layers": mid_late_layers
            },   
            "PATCH_TO_LAST_ROW_LATE_LAYERS": {
                "from": list(range(start_end_indices[0][0], start_end_indices[0][1]-24)),
                "to": list(range(start_end_indices[0][1]-24, start_end_indices[0][1])),
                "layers": late_layers
            },   
        }
        experiment_keys.update(experiments.keys())


        # Run attention blocking experiments
        results[str(img_id)]["attn_block_results"] = {}
        for exp_name, exp_config in experiments.items():
            if exp_name in results[str(img_id)]["attn_block_results"]:
                continue
            attn_block_dict = get_attention_block_dict(exp_config["from"], exp_config["to"], exp_config["layers"])
            result = run_experiment(model, image, prompt, class_name, attn_block_dict)
            results[str(img_id)]["attn_block_results"][exp_name] = result

        # Save intermediate results
        with open(results_file, "w") as f:
            json.dump(results, f)

    # Print summary results
    print("Baseline accuracy:", sum(1 for img in results.values() if img["baseline"]["is_correct"]) / len(results))

    # Print attention blocking results
    print("\nAttention Blocking Results:")

    # for exp_name in experiment_keys:
    for exp_name in list(results.values())[0]["attn_block_results"]:
        correct_count = sum(1 for img in results.values() if "attn_block_results" in img and img["attn_block_results"][exp_name]["is_correct"])
        accuracy = correct_count / len(results)
        
        # Calculate average probability drop
        prob_drops = []
        for img in results.values():
            if "attn_block_results" in img and exp_name in img["attn_block_results"]:
                baseline_prob = img["baseline"]["generated_token_prob"]
                exp_prob = img["attn_block_results"][exp_name]["generated_token_prob"]
                prob_drops.append(baseline_prob - exp_prob)
        
        avg_prob_drop = sum(prob_drops) / len(prob_drops) if prob_drops else 0
        
        print(f">>> {exp_name}:")
        print(f"    Accuracy: {accuracy:.2f}")
        print(f"    Avg Probability Drop: {avg_prob_drop:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process images with clean questions.")
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use (e.g., cuda:0, cpu)')
    parser.add_argument('--datatype', type=str, default='val2017', help='Data type for COCO dataset')
    parser.add_argument('--results_file', type=str, required=True, help='File to save results')
    parser.add_argument('--clean_questions_file', type=str, required=True, help='JSON file containing clean questions')

    args = parser.parse_args()

    main(args.device, args.datatype, args.results_file, args.clean_questions_file)