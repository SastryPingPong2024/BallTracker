import os
import subprocess
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    
    # Parse the output and convert to integers
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return gpu_memory

def set_available_gpu(env):
    try:
        gpu_memory_map = get_gpu_memory_map()
        if gpu_memory_map:
            # Select the GPU with the most free memory
            selected_gpu = gpu_memory_map.index(max(gpu_memory_map))
            env["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
            return selected_gpu
        else:
            print("No GPUs available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("nvidia-smi is not available. Make sure CUDA is installed and you have NVIDIA GPUs.")

def process_match(match_folder, root_dir):
    match_number = match_folder.split('match')[1]

    # Set the CUDA_VISIBLE_DEVICES environment variable
    env = os.environ.copy()
    gpu_id = set_available_gpu(env)

    command = [
        "python", "predict.py",
        f"--video_file={os.path.join(root_dir, match_folder)}",
        "--tracknet_file=finetune/TrackNet_best.pt",
        "--inpaintnet_file=finetune/InpaintNet_best.pt",
        f"--save_dir=outputs/{match_folder}",
        "--cuda=0"  # Always use cuda:0 as it's the only visible device for this process
    ]

    print(f"Processing {match_folder} on GPU {gpu_id}")
    subprocess.run(command, check=True, env=env)
    print(f"Finished processing {match_folder}")

def main(root_dir):
    match_folders = [f for f in os.listdir(root_dir) if f.startswith('match')]
    match_folders.sort(key=lambda x: int(x.split('match')[1]))

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, match_folder in enumerate(match_folders):
            future = executor.submit(process_match, match_folder, root_dir)
            futures.append(future)
            time.sleep(10)

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process match folders in parallel using multiple GPUs")
    parser.add_argument("root_dir", help="Root directory containing match folders")
    args = parser.parse_args()

    main(args.root_dir)