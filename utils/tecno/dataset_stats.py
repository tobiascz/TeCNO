import numpy as np
from tqdm import tqdm
import time
from pathlib import Path


def convert_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def count_frames(path, ids, file_format=".png"):
    img_base_path = Path(path)
    video_lengths = {}
    for video_name in tqdm(ids):
        img_path_for_vid = img_base_path / video_name
        img_list = sorted(img_path_for_vid.glob(f'*{file_format}'))
        img_list = [str(i.relative_to(img_base_path)) for i in img_list]
        video_lengths[video_name] = len(img_list)
        print(f"{video_name} length: {len(img_list)}")
    return video_lengths


def print_stats(video_len_array, fps=25):
    video_len_array_in_seconds = [x / fps for x in video_len_array]
    video_len_array_in_seconds = np.asarray(video_len_array_in_seconds)
    mean_time = np.mean(video_len_array_in_seconds)
    median_time = np.median(video_len_array_in_seconds)
    max_time = np.max(video_len_array_in_seconds)
    min_time = np.min(video_len_array_in_seconds)
    print(f"Mean time per video: {convert_time(mean_time)}")
    print(f"Median time per video: {convert_time(median_time)}")
    print(f"Max time per video: {convert_time(max_time)}")
    print(f"Min time per video: {convert_time(min_time)}")


if __name__ == "__main__":
    dataset = "cholec"

    if dataset == "cholec":
        print(f"stats for cholec....\n\n")
        root_dir = "/CHOLECDATASET"
        pat_ids = [f"video{i:02d}" for i in range(1, 81)]
        video_lengths = count_frames(root_dir, pat_ids, file_format=".png")
        video_len_array = [v for k, v in video_lengths.items()]
        print_stats(video_len_array, fps=25)
