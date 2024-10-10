import argparse
import json
import os
import torch
from tqdm import tqdm
from torch.nn import functional as F
from PIL import Image
import yaml

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HookedLVLM import HookedLVLM
from ImageDatasets import COCOImageDataset
from utils import correct_annotations_for_crop, get_object_patch_indices, get_register_indices, get_random_indices
from InputsEmbeds import InputsEmbeds

def format_and_compare_answers(model, prompt, answer, class_name):
    generated_answer = answer.strip().lower()
    tokens = model.processor.tokenizer.tokenize(class_name.lower())
    first_token = tokens[0].strip().lstrip('Ġ▁')
    return first_token.lower().startswith(generated_answer.lower())

def run_experiment(model, image, prompt, class_name, ablate_indices=None, replacement_tensor=None):
    with torch.no_grad():
        if ablate_indices is not None and replacement_tensor is not None:
            with model.ablate_inputs(indices=ablate_indices, replacement_tensor=replacement_tensor):
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

def integrated_gradients(model, image, prompt, image_mean_tensor, model_id, steps=50):
    with torch.no_grad():
        input_embeds = model.get_text_model_in(image, prompt)
        _, _, img_start_end_indices = InputsEmbeds(model.processor.tokenizer, input_embeds, prompt, model_id).get_img_and_text_embed()
        img_start, img_end = img_start_end_indices[0]
        baseline_embeds = input_embeds.clone()

        replacement_tensor = image_mean_tensor.to(baseline_embeds.dtype).to(baseline_embeds.device)
        replacement_tensor.unsqueeze(0).expand((img_end-img_start+1), -1)

        baseline_embeds[:, img_start:img_end+1, :] = replacement_tensor

    diff = input_embeds - baseline_embeds
    integrated_grads = torch.zeros_like(input_embeds)
    
    for alpha in torch.linspace(0, 1, steps):
        interpolated_embeds = baseline_embeds + alpha * diff
        interpolated_embeds.requires_grad_(True)
        
        outputs = model.model(inputs_embeds=interpolated_embeds)
        logits = outputs.logits
        
        yes_logit = logits[0, -1, model.processor.tokenizer.encode("Yes")[0]]
        
        model.model.zero_grad()
        yes_logit.backward(retain_graph=True)
        
        integrated_grads += interpolated_embeds.grad
    
    integrated_grads *= diff / steps
    
    return integrated_grads

def get_high_gradient_indices(model, image, prompt, class_name, image_mean_tensor, model_id, steps=50):
    integrated_grads = integrated_gradients(model, image, prompt, image_mean_tensor, model_id, steps)
    
    token_importance = integrated_grads.abs().sum(dim=-1).squeeze()
    sorted_indices = torch.argsort(token_importance, descending=True)
    
    return sorted_indices.tolist()

def main(device, data_dir, data_type, clean_questions_file, results_file, mean_tensor_loc, zero_ablation):
    # Load and Filter COCO Dataset
    ann_file = '{}/annotations/instances_{}.json'.format(data_dir, data_type)

    cfg = {
        "zero_ablation": zero_ablation,
    }

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

    ds = COCOImageDataset(data_dir=data_dir, data_type=data_type, ann_file=ann_file, filter_fn=filter_fn)

    # Load Model
    model_id = "xtuner/llava-phi-3-mini-hf"
    model = HookedLVLM(model_id=model_id, device=device, quantize=True, quantize_type="fp16")
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

    # Load image mean tensor for ablation
    image_mean_tensor = torch.load(mean_tensor_loc)

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
        prompt = f"<|user|>\n<image>{text_question}<|end|>\n<|assistant|>\nIt is a"

        # Run baseline experiment
        baseline_result = run_experiment(model, image, prompt, class_name)
        results[str(img_id)]["baseline"] = baseline_result

        # Skip experiments if baseline is incorrect
        if not baseline_result["is_correct"]:
            continue

        # Prepare experiments
        activations = model.get_text_model_in(image, prompt)
        input_embeds = InputsEmbeds(model.processor.tokenizer, activations, prompt, model_id)
        img_embeds, text_embeds, start_end_indices = input_embeds.get_img_and_text_embed()

        experiments = {
            "register_ablation": {"indices": get_register_indices(img_embeds, text_embeds, start_end_indices)},
            "patch_ablation": {"indices": get_object_patch_indices(image, ann, start_end_indices, buffer=0)},
            "patch_plus_one_ablation": {"indices": get_object_patch_indices(image, ann, start_end_indices, buffer=1)},
            "patch_plus_two_ablation": {"indices": get_object_patch_indices(image, ann, start_end_indices, buffer=2)},
        }

        # Add gradient ablation experiments
        gradient_indices = get_high_gradient_indices(model, image, prompt, class_name, image_mean_tensor, model_id)
        for n in [5, 10, 20, 40, 60, 100, 250]:
            experiments[f"gradient_ablation_{n}"] = {"indices": gradient_indices[:n]}

        # Add random ablation experiments
        for n in [5, 10, 20, 40, 60, 100, 250]:
            experiments[f"random_ablation_{n}"] = {"indices": get_random_indices(start_end_indices, n)}

        # Run experiments
        results[str(img_id)]["experiments"] = {}
        for exp_name, exp_config in experiments.items():
            result = run_experiment(model, image, prompt, class_name, 
                                    ablate_indices=exp_config["indices"], 
                                    replacement_tensor=image_mean_tensor)
            results[str(img_id)]["experiments"][exp_name] = result

        # Save intermediate results
        with open(results_file, "w") as f:
            json.dump(results, f)

    # Print summary results
    print("Baseline accuracy:", sum(1 for img in results.values() if img["baseline"]["is_correct"]) / len(results))

    # Print experiment results
    print("\nExperiment Results:")
    for exp_name in experiments.keys():
        correct_count = sum(1 for img in results.values() if "experiments" in img and img["experiments"][exp_name]["is_correct"])
        accuracy = correct_count / len(results)
        
        # Calculate average probability drop
        prob_drops = []
        for img in results.values():
            if "experiments" in img and exp_name in img["experiments"]:
                baseline_prob = img["baseline"]["generated_token_prob"]
                exp_prob = img["experiments"][exp_name]["generated_token_prob"]
                prob_drops.append(baseline_prob - exp_prob)
        
        avg_prob_drop = sum(prob_drops) / len(prob_drops) if prob_drops else 0
        
        print(f">>> {exp_name}:")
        print(f"    Accuracy: {accuracy:.2f}")
        print(f"    Avg Probability Drop: {avg_prob_drop:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run comprehensive ablation experiments on clean dataset with Phi model.")
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use (e.g., cuda:0, cpu)')
    parser.add_argument('--datatype', type=str, default='val2017', help='Data type for COCO dataset')
    parser.add_argument('--results_file', type=str, required=True, help='File to save results')
    parser.add_argument('--clean_questions_file', type=str, required=True, help='JSON file containing clean questions')

    args = parser.parse_args()

    config_file = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    data_dir = config['data_dir']

    main(args.device, data_dir, args.datatype, args.clean_questions_file, args.results_file, args.mean_tensor, args.zero_ablation)