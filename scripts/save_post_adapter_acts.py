import argparse
from PIL import Image
import os
import torch
import pickle
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from pathlib import Path

class ImageProcessor:
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf", device="cuda:0"):
        self.device = device
        self.model_name = model_name
        self.model, self.image_processor = self._initialize_model_and_processor()

    def _initialize_model_and_processor(self):
        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True,
            device_map={
                'vision_tower': int(self.device[-1]),
                'multi_modal_projector': int(self.device[-1]),
                'language_model': 'cpu'
            },
        )
        image_processor = AutoProcessor.from_pretrained(self.model_name).image_processor
        return model, image_processor

    def process_images(self, image_dir, save_path, batch_size=32):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        filenames = [f for f in os.listdir(image_dir) if f.endswith(".JPEG")]
        results_dict = {}

        for i in tqdm(range(0, len(filenames), batch_size), desc="Processing batches"):
            batch_files = filenames[i:i+batch_size]
            batch_images = [Image.open(os.path.join(image_dir, f)).convert('RGB') for f in batch_files]

            with torch.no_grad():
                b_out = self._process_batch(batch_images)

            # Save the entire batch tensor
            batch_tensor_filename = f"batch_{i // batch_size}.pt"
            torch.save(b_out, save_path / batch_tensor_filename)

            # Update results_dict with file mappings
            for j, file in enumerate(batch_files):
                results_dict[file[:-5]] = (batch_tensor_filename, j)

        with open(save_path / "results_dict.pkl", "wb") as f:
            pickle.dump(results_dict, f)

        print(f"Processing completed. Results saved to {save_path}.")

    def _process_batch(self, batch_images):
        inputs = self.image_processor(images=batch_images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)

        a_out = self.model.vision_tower(pixel_values, output_hidden_states=True)
        a_out = a_out.hidden_states[-2][:, 1:] # The second last layer is what LLAVA actually uses. Removing the first token as it is [CLS].
        return self.model.multi_modal_projector(a_out)

def main():
    parser = argparse.ArgumentParser(description="Process images in batches and save outputs.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images to process")
    parser.add_argument("--save_path", type=str, required=True, help="File path to save the processing results")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of images to process in a batch")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:n to use")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf", help="Name of the model to use")

    args = parser.parse_args()

    processor = ImageProcessor(args.model_name, args.device)
    processor.process_images(args.image_dir, args.save_path, args.batch_size)

if __name__ == "__main__":
    main()