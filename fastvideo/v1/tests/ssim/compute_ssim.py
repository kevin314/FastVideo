import torch
import numpy as np
import os
from pytorch_msssim import ssim, ms_ssim
from torchvision.io import read_video

def compute_video_ssim_torchvision(video1_path, video2_path, use_ms_ssim=True):
    print(f"Computing SSIM between {video1_path} and {video2_path}...")

    # Load videos using torchvision
    frames1, _, _ = read_video(video1_path, pts_unit='sec')
    frames2, _, _ = read_video(video2_path, pts_unit='sec')
    
    # Ensure same number of frames
    min_frames = min(frames1.shape[0], frames2.shape[0])
    frames1 = frames1[:min_frames]
    frames2 = frames2[:min_frames]
    
    ssim_values = []
    
    # Process in batches to avoid memory issues
    batch_size = 8  # Adjust based on your GPU memory
    for i in range(0, min_frames, batch_size):
        batch1 = frames1[i:i+batch_size].float() / 255.0
        batch2 = frames2[i:i+batch_size].float() / 255.0
        
        # Convert from [B, H, W, C] to [B, C, H, W]
        batch1 = batch1.permute(0, 3, 1, 2)
        batch2 = batch2.permute(0, 3, 1, 2)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            batch1 = batch1.cuda()
            batch2 = batch2.cuda()
        
        # Process each sample in the batch individually
        for j in range(batch1.shape[0]):
            img1 = batch1[j:j+1]  # Keep batch dimension
            img2 = batch2[j:j+1]  # Keep batch dimension
            
            # Compute SSIM or MS-SSIM
            with torch.no_grad():
                if use_ms_ssim:
                    value = ms_ssim(img1, img2, data_range=1.0)
                else:
                    value = ssim(img1, img2, data_range=1.0)
                
                ssim_values.append(value.item())  # Convert tensor to scalar

    if ssim_values:
        mean_ssim = np.mean(ssim_values)
        print(f"Mean SSIM: {mean_ssim}")
        return mean_ssim
    else:
        print('No SSIM values calculated')
        return 0

def compare_folders(reference_folder, generated_folder, use_ms_ssim=True):
    """
    Compare videos with the same filename between reference_folder and generated_folder
    """
    # Get the list of video files in the reference folder
    reference_videos = [f for f in os.listdir(reference_folder) 
                        if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    results = {}
    
    for video_name in reference_videos:
        ref_path = os.path.join(reference_folder, video_name)
        gen_path = os.path.join(generated_folder, video_name)
        
        # Check if the corresponding generated video exists
        if os.path.exists(gen_path):
            print(f"\nComparing {video_name}...")
            try:
                ssim_value = compute_video_ssim_torchvision(ref_path, gen_path, use_ms_ssim)
                results[video_name] = ssim_value
            except Exception as e:
                print(f"Error comparing {video_name}: {e}")
                results[video_name] = None
        else:
            print(f"\nSkipping {video_name} - no matching file in generated folder")
    
    return results

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths to reference and generated video folders
    reference_folder = os.path.join(script_dir, 'reference_videos')
    generated_folder = os.path.join(script_dir, 'generated_videos')
    
    # Check if folders exist
    if not os.path.exists(reference_folder):
        print(f"ERROR: Reference folder {reference_folder} does not exist!")
        exit(1)
    
    if not os.path.exists(generated_folder):
        print(f"ERROR: Generated folder {generated_folder} does not exist!")
        exit(1)
    
    # Compare videos in the folders
    print(f"Comparing videos between {reference_folder} and {generated_folder}")
    results = compare_folders(reference_folder, generated_folder)
    
    # Print summary of results
    print("\n===== SSIM Results Summary =====")
    for video_name, ssim_value in results.items():
        if ssim_value is not None:
            print(f"{video_name}: {ssim_value:.4f}")
        else:
            print(f"{video_name}: Error during comparison")
    
    # Calculate average SSIM across all videos
    valid_ssims = [v for v in results.values() if v is not None]
    if valid_ssims:
        avg_ssim = np.mean(valid_ssims)
        print(f"\nAverage SSIM across all videos: {avg_ssim:.4f}")
    else:
        print("\nNo valid SSIM values to average")
