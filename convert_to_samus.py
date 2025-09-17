import os
import shutil
import argparse
from pathlib import Path


def ensure_dir(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def convert_split_file(split_file, split_name, new_mainpatient_dir, orig_root, img_dir, label_dir, subtask_name, class_id):
    """
    Converts original split text files to SAMUS train/val/test format.
    """
    samus_lines = []
    with open(split_file, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines:
        frame_path, label = line.split(":")
        frame_name = os.path.basename(frame_path).replace(".png", "")

        # For train/val -> "class_id/SubtaskName/frame_name"
        if split_name in ["train", "val"]:
            samus_lines.append(f"{class_id}/{subtask_name}/{frame_name}")
        else:  # For test -> "SubtaskName/frame_name"
            samus_lines.append(f"{subtask_name}/{frame_name}")

        # Copy frame to SAMUS img/ folder
        src_frame = orig_root / "frames" / os.path.basename(frame_path)
        dst_frame = img_dir / os.path.basename(frame_path)
        shutil.copy2(src_frame, dst_frame)

        # Copy mask to SAMUS label/ folder
        src_mask = orig_root / "masks" / os.path.basename(frame_path)
        dst_mask = label_dir / os.path.basename(frame_path)
        shutil.copy2(src_mask, dst_mask)

    # Write the new split file
    new_split_file = new_mainpatient_dir / f"{split_name}.txt"
    with open(new_split_file, "w") as f:
        f.write("\n".join(samus_lines))
    print(f"[INFO] Created {new_split_file} with {len(samus_lines)} entries.")


def main():
    parser = argparse.ArgumentParser(description="Convert custom dataset to SAMUS format")
    parser.add_argument("--orig_root", type=str, required=True,
                        help="Path to the original dataset root directory (containing frames/, masks/, data_splits/).")
    parser.add_argument("--new_root", type=str, required=True,
                        help="Path to the new SAMUS dataset root directory to be created.")
    parser.add_argument("--subtask_name", type=str, default="Niches",
                        help="Name of the dataset subfolder in SAMUS format (default: Niches).")
    parser.add_argument("--class_id", type=int, default=1,
                        help="Class ID for positive class (default: 1).")

    args = parser.parse_args()

    orig_root = Path(args.orig_root).resolve()
    new_root = Path(args.new_root).resolve()
    subtask_name = args.subtask_name
    class_id = args.class_id

    print(f"[INFO] Original dataset: {orig_root}")
    print(f"[INFO] Creating new SAMUS dataset: {new_root}")

    # Create necessary directories
    img_dir = new_root / subtask_name / "img"
    label_dir = new_root / subtask_name / "label"
    mainpatient_dir = new_root / "MainPatient"

    for d in [img_dir, label_dir, mainpatient_dir]:
        ensure_dir(d)

    # Process each split file
    splits = ["train", "val", "test"]
    for split in splits:
        split_file = orig_root / "data_splits" / f"{split}.txt"
        if not split_file.exists():
            print(f"[WARN] {split_file} does not exist, skipping...")
            continue
        convert_split_file(split_file, split, mainpatient_dir,
                           orig_root, img_dir, label_dir,
                           subtask_name, class_id)

    # Create class.json
    class_json_path = mainpatient_dir / "class.json"
    with open(class_json_path, "w") as f:
        f.write(f'{{\n    "{subtask_name}": 2\n}}\n')
    print(f"[INFO] Created class.json at {class_json_path}")

    print("\n[INFO] Conversion complete!")
    print(f"[INFO] Final SAMUS dataset structure at {new_root}")


if __name__ == "__main__":
    main()


"""
EXAMPLE COMMAND:

python convert_to_samus.py \
  --orig_root /net/beegfs/groups/gaia/niche_segmentation_storage/datasets/niches_15_9_sbaf0_pnc100 \
  --new_root /net/beegfs/groups/gaia/niche_segmentation_storage/datasets/niches_15_9_sbaf0_pnc100_SAMUS \
  --subtask_name Niches \
  --class_id 1

"""
