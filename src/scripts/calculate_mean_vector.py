import os
import torch
import argparse
from tqdm import tqdm

def calculate_mean_vector(directory, device):
    total_sum = None
    total_count = 0

    files = [filename for filename in os.listdir(directory) if filename.endswith('.pt')]

    for filename in tqdm(files):
        if filename.endswith('.pt'):
            file_path = os.path.join(directory, filename)
            
            # Load the tensor and move it to the specified device
            tensor = torch.load(file_path).to(device)
            
            # Compute the sum along the first two dimensions
            file_sum = tensor.sum(dim=(0, 1))
            
            # Update the total sum
            if total_sum is None:
                total_sum = file_sum
            else:
                total_sum += file_sum
            
            # Update the total count
            total_count += tensor.shape[0] * tensor.shape[1]
            
            # Free up memory
            del tensor
            if device == 'cuda':
                torch.cuda.empty_cache()

    # Calculate the mean
    mean_vector = total_sum / total_count

    return mean_vector

def main():
    parser = argparse.ArgumentParser(description='Calculate mean vector across multiple .pt files.')
    parser.add_argument('directory', type=str, help='Directory containing the .pt files')
    parser.add_argument('--device', type=str, default='cpu', 
                        help='Device to use for calculations (default: cpu)')
    
    args = parser.parse_args()

    # Check if CUDA is available when 'cuda' is specified
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        args.device = 'cpu'

    print(f"Using device: {args.device}")

    # Calculate the mean vector
    mean_vector = calculate_mean_vector(args.directory, args.device)

    # Save the mean vector
    output_path = os.path.join(args.directory, 'mean_vector.pt')
    torch.save(mean_vector.cpu(), output_path)  # Save to CPU to ensure compatibility

    print(f"Mean vector saved to {output_path}")

if __name__ == "__main__":
    main()