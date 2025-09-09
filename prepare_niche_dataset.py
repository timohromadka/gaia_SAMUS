import os
import json
import argparse
import random
import cv2
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Niche Dataset for SAMUS")
    parser.add_argument("--jsons_folder", type=str, required=True, help="Folder containing annotation JSON files")
    parser.add_argument("--videos_folder", type=str, required=True, help="Folder containing raw videos (.mp4)")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder for prepared dataset")
    parser.add_argument("--percent_niche_cases", type=float, required=True, help="Fraction of positive frames (0-1)")
    parser.add_argument("--max_samples", type=int, required=True, help="Maximum total frames to include")
    return parser.parse_args()

def extract_polygons_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    data_unit = list(data[0]["data_units"].values())[0]
    width, height = data_unit["width"], data_unit["height"]
    labels = data_unit.get("labels", {})

    frame_polygons = {}
    for frame_str, frame_data in labels.items():
        frame_num = int(frame_str)
        objects = frame_data.get("objects", [])
        polygons_for_frame = []

        for obj in objects:
            # Some JSONs may store polygon coordinates in "polygons" or "polygon"
            obj_polygons = obj.get("polygons", [])
            if not obj_polygons and "polygon" in obj:
                # convert "polygon" dict to a list of [x0, y0, x1, y1, ...]
                coords = []
                for k in sorted(obj["polygon"].keys(), key=int):
                    coords.extend([obj["polygon"][k]["x"], obj["polygon"][k]["y"]])
                obj_polygons = [coords]
            # Scale normalized coords to pixels
            pixel_polygons = []
            for poly in obj_polygons:
                # poly may be a list of lists
                if isinstance(poly[0], list):
                    # flatten
                    poly = [coord for sublist in poly for coord in sublist]

                pixel_poly = []
                for i in range(0, len(poly), 2):
                    x_px = int(poly[i] * width)
                    y_px = int(poly[i+1] * height)
                    pixel_poly.extend([x_px, y_px])
                pixel_polygons.append(pixel_poly)

            polygons_for_frame.extend(pixel_polygons)

        if polygons_for_frame:
            frame_polygons[frame_num] = polygons_for_frame

    return frame_polygons, width, height


def extract_frames_and_masks(video_path, polygons_dict, output_img_dir, output_label_dir, prefix):
    """Extract frames and masks from a single video."""
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    positive_frames = []
    negative_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_name = f"{prefix}_frame{frame_idx:04d}.png"
        img_path = os.path.join(output_img_dir, frame_name)
        mask_path = os.path.join(output_label_dir, frame_name)

        # Save image
        cv2.imwrite(img_path, frame)

        # Create mask
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        if frame_idx in polygons_dict:
            for poly in polygons_dict[frame_idx]:
                poly_array = np.array(poly, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(mask, [poly_array], 255)  # works now
            cv2.imwrite(mask_path, mask)
            positive_frames.append(frame_name)

        else:
            cv2.imwrite(mask_path, mask)
            negative_frames.append(frame_name)

        frame_idx += 1

    cap.release()
    return positive_frames, negative_frames

def prepare_dataset(jsons_folder, videos_folder, output_folder, percent_niche_cases, max_samples):
    img_dir = os.path.join(output_folder, "img")
    label_dir = os.path.join(output_folder, "label")
    mainpatient_dir = os.path.join(output_folder, "MainPatient")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(mainpatient_dir, exist_ok=True)

    # Match videos with JSON files using metadata inside JSON
    video_files_dict = {f.lower(): f for f in os.listdir(videos_folder) if f.lower().endswith(".mp4")}


    json_video_pairs = []
    for json_file in os.listdir(jsons_folder):
        if not json_file.endswith(".json"):
            continue
        json_path = os.path.join(jsons_folder, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
        try:
            # Extract data_title from metadata
            data_units = list(data[0]["data_units"].values())[0]
            data_title = data_units["data_title"].lower()  # e.g., "VU1161_sag_cropped.mp4"
        except KeyError:
            continue

        if data_title in video_files_dict:
            video_file_actual = video_files_dict[data_title]
            video_path = os.path.join(videos_folder, video_file_actual)
            json_video_pairs.append((json_path, video_path))

    if not json_video_pairs:
        raise ValueError("No matching videos and JSON annotations found!")

    print(f"Found {len(json_video_pairs)} videos with matching annotations.")


    all_positive_frames = []
    all_negative_frames = []

    for json_path, video_path in tqdm(json_video_pairs, desc="Processing videos"):
        # Extract prefix from video filename without extension
        prefix = os.path.splitext(os.path.basename(video_path))[0]

        polygons_dict, width, height = extract_polygons_from_json(json_path)
        pos_frames, neg_frames = extract_frames_and_masks(video_path, polygons_dict, img_dir, label_dir, prefix=prefix)

        # Store with full path and class ID
        all_positive_frames.extend([f"0/{os.path.basename(output_folder)}/{fname}" for fname in pos_frames])
        all_negative_frames.extend([f"0/{os.path.basename(output_folder)}/{fname}" for fname in neg_frames])

    # Balance positive and negative frames
    total_pos = len(all_positive_frames)
    desired_pos = int(max_samples * percent_niche_cases)

    if total_pos < desired_pos:
        print(f"Not enough positive frames ({total_pos}) to satisfy desired fraction. Reducing dataset size.")
        desired_pos = total_pos
        max_samples = int(desired_pos / percent_niche_cases)

    # Final sample sizes
    desired_neg = max_samples - desired_pos

    selected_pos = random.sample(all_positive_frames, desired_pos)
    selected_neg = random.sample(all_negative_frames, desired_neg)

    final_frames = selected_pos + selected_neg
    random.shuffle(final_frames)

    # Split into train/val/test
    train_end = int(0.7 * len(final_frames))
    val_end = int(0.85 * len(final_frames))

    train_frames = final_frames[:train_end]
    val_frames = final_frames[train_end:val_end]
    test_frames = final_frames[val_end:]

    # Write to txt files
    with open(os.path.join(mainpatient_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_frames))

    with open(os.path.join(mainpatient_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_frames))

    with open(os.path.join(mainpatient_dir, "test.txt"), "w") as f:
        f.write("\n".join(test_frames))

    print(f"Dataset prepared successfully in {output_folder}")
    print(f"Train: {len(train_frames)}, Val: {len(val_frames)}, Test: {len(test_frames)}")

def main():
    args = parse_args()
    prepare_dataset(
        args.jsons_folder,
        args.videos_folder,
        args.output_folder,
        args.percent_niche_cases,
        args.max_samples
    )

if __name__ == "__main__":
    main()

"""
python prepare_niche_dataset.py \
    --jsons_folder /home/P098475/gaia_SAMUS/raw_jsons \
    --videos_folder /home/P098475/gaia_SAMUS/raw_videos \
    --output_folder /home/P098475/gaia_SAMUS/temp_samus_dataset_format_test \
    --percent_niche_cases 0.6 \
    --max_samples 100041231

"""