import os
import sys
import argparse
from PIL import Image
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', '..', 'src')
sys.path.append(src_path)

from src.HookedLVLM import HookedLVLM
from src.lvlm_lens import create_interactive_logit_lens

def is_image_file(filename):
    valid_extensions = ('.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG')
    return filename.lower().endswith(valid_extensions)

def process_images(image_folder, save_folder, device, quantize_type, num_images):
    # Import Model
    model = HookedLVLM(device=device, quantize=True, quantize_type=quantize_type)

    # Load components needed for logit lens
    norm = model.model.language_model.model.norm
    lm_head = model.model.language_model.lm_head
    tokenizer = model.processor.tokenizer
    model_name = model.model.config._name_or_path.split("/")[-1]

    # Load images
    image_files = [f for f in os.listdir(image_folder) if is_image_file(f)]
    if num_images:
        image_files = image_files[:num_images]
    
    image_paths = [os.path.join(image_folder, f) for f in image_files]
    images = {}
    for image_path in image_paths:
        try:
            image = Image.open(image_path)
            images[image_path] = image
        except IOError:
            print(f"Could not open image file: {image_path}")

    # Run forward pass
    for image_path, image in tqdm(images.items()):
        text_question = "Describe the image."
        prompt = f"USER: <image>\n{text_question} ASSISTANT:"

        hidden_states = model.forward(image, prompt, output_hidden_states=True).hidden_states
        create_interactive_logit_lens(hidden_states, norm, lm_head, tokenizer, image, model_name, image_path, prompt, save_folder)

def main():
    parser = argparse.ArgumentParser(description="Process images using HookedLVLM model")
    parser.add_argument("--image_folder", required=True, help="Path to the folder containing images")
    parser.add_argument("--save_folder", required=True, help="Path to save the results")
    parser.add_argument("--device", default="cuda:0", help="Device to run the model on")
    parser.add_argument("--quantize_type", default="fp16", help="Quantization type")
    parser.add_argument("--num_images", type=int, help="Number of images to process (optional)")

    args = parser.parse_args()

    process_images(args.image_folder, args.save_folder, args.device, args.quantize_type, args.num_images)

if __name__ == "__main__":
    main()