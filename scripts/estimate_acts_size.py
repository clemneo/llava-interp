"""
Based off save_post_adapter_acts.py. 
Basically runs 1 image through and 
estimates the total size of all activations.
"""

import argparse
import os
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import numpy as np

def estimate_total_size(image_dir, sample_size=5):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # Initialize the model and processor
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        device_map=device,
    )
    image_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf").image_processor

    # Prepare model components for processing
    vision_tower = model.vision_tower
    multi_modal_projector = model.multi_modal_projector

    # List all JPEG files in the directory
    filenames = [f for f in os.listdir(image_dir) if f.endswith(".JPEG") or f.endswith(".png")]
    total_images = len(filenames)

    # Randomly select a sample of images
    sample_files = np.random.choice(filenames, size=min(sample_size, total_images), replace=False)
    print(sample_files)

    total_size = 0
    for filename in sample_files:
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path).convert('RGB')

        # Prepare the image for the model
        inputs = image_processor(images=image, return_tensors="pt")
        # pixel_values = inputs['pixel_values'].to(torch.float16).to(device)
        pixel_values = inputs['pixel_values'].to(device)

        with torch.no_grad():
            a_out = vision_tower(pixel_values, output_hidden_states=True)
            a_out = a_out.hidden_states[-2]
            a_out = a_out[:, 1:]
            b_out = multi_modal_projector(a_out)

        print(b_out.shape)
        print(b_out)
        # Calculate size of b_out in bytes
        size_bytes = b_out.element_size() * b_out.nelement()
        total_size += size_bytes

    print(size_bytes)
    # Calculate average size per image
    avg_size = total_size / len(sample_files)

    # Estimate total size for all images
    estimated_total_size = avg_size * total_images

    return estimated_total_size

def main():
    parser = argparse.ArgumentParser(description="Estimate total size of processed images.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images to process")
    parser.add_argument("--sample_size", type=int, default=1, help="Number of images to sample for estimation")

    args = parser.parse_args()

    estimated_size = estimate_total_size(args.image_dir, args.sample_size)

    print(f"Estimated total size: {estimated_size / (1024**3):.2f} GB")

if __name__ == "__main__":
    main()