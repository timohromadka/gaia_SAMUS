import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
import random
from utils.config import get_config
from utils.evaluation import get_eval
from models.model_dict import get_model
from utils.data_us import JointTransform2D, ImageToImage2D
from utils.loss_functions.sam_loss import get_criterion
from thop import profile
from tqdm import tqdm

def extract_frames(video_path, temp_frame_dir):
    """Extract frames from a video using cv2 and save to temp dir."""
    os.makedirs(temp_frame_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_paths = []

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(temp_frame_dir, f"frame_{count:05d}.png")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        count += 1
    cap.release()
    return frame_paths, fps

def overlay_mask_on_frame(frame, mask):
    """Overlay segmentation mask (binary) on a frame in red."""
    overlay = frame.copy()
    mask_red = np.zeros_like(frame)
    mask_red[:, :, 2] = (mask * 255).astype(np.uint8)  # red channel

    # Blend original frame and red mask
    blended = cv2.addWeighted(overlay, 1.0, mask_red, 0.5, 0)
    return blended

def frames_to_video(frame_paths, output_path, fps):
    """Combine frames back into a video."""
    if len(frame_paths) == 0:
        raise ValueError("No frames to combine into a video.")

    first_frame = cv2.imread(frame_paths[0])
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        out.write(frame)
    out.release()

def run_inference_on_video(model, device, video_path, output_video_path, args, opt):
    """Run inference on all frames of a video and save an output video."""
    temp_frame_dir = os.path.join("temp_frames", os.path.basename(video_path).split('.')[0])
    os.makedirs(temp_frame_dir, exist_ok=True)

    # Step 1: Extract frames
    frame_paths, fps = extract_frames(video_path, temp_frame_dir)

    processed_frame_paths = []
    model.eval()

    # Step 2: Run inference on each frame
    for frame_path in tqdm(frame_paths, desc=f"Processing {os.path.basename(video_path)}"):
        frame = cv2.imread(frame_path)
        original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(original_frame, (args.encoder_input_size, args.encoder_input_size))
        input_tensor = torch.from_numpy(resized_frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        input_tensor = input_tensor.to(device)

        # Assuming your model takes (image, points) as input
        dummy_points = (
            torch.tensor([[[1, 2]]]).float().to(device),
            torch.tensor([[1]]).float().to(device)
        )

        with torch.no_grad():
            output = model(input_tensor, dummy_points)

        # Assume output is [B, 1, H, W] binary mask
        mask_tensor = output["masks"]  # Shape: [B, 1, H, W]
        mask = torch.sigmoid(mask_tensor).squeeze().cpu().numpy()  # Convert to numpy
        mask = (mask > 0.5).astype(np.uint8)  # Binarize at threshold 0.5
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))


        # Step 3: Overlay mask on frame
        blended_frame = overlay_mask_on_frame(frame, mask_resized)

        output_frame_path = os.path.join(temp_frame_dir, f"processed_{os.path.basename(frame_path)}")
        cv2.imwrite(output_frame_path, blended_frame)
        processed_frame_paths.append(output_frame_path)

    # Step 4: Combine processed frames into a video
    frames_to_video(processed_frame_paths, output_video_path, fps)

    # Optional: cleanup temporary frames
    # import shutil
    # shutil.rmtree(temp_frame_dir)

    # Optional: cleanup temporary frames
    # import shutil
    # shutil.rmtree(temp_frame_dir)

def main():
    # =========================================== Parameters ==================================================
    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='SAMUS', type=str, help='Model name: SAM, SAMFull, SAMUS...')
    parser.add_argument('--encoder_input_size', type=int, default=256, help='Image size of encoder input')
    parser.add_argument('--low_image_size', type=int, default=128, help='Image embedding size')
    parser.add_argument('--task', default='BUSI', help='Task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='ViT model for image encoder of SAM')
    parser.add_argument('--sam_ckpt', type=str, default='/net/beegfs/groups/gaia/gaia_SAMUS_storage/checkpoints/NICHES/sam_vit_b_01ec64.pth', help='Pretrained SAM checkpoint')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--n_gpu', type=int, default=1, help='Total GPUs')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--warmup', type=bool, default=False)
    parser.add_argument('--warmup_period', type=int, default=250)
    parser.add_argument('--keep_log', type=bool, default=False)

    # IF doing inference ONLY on videos
    parser.add_argument('--input_mp4_directory', type=str, default=None,
                        help='Directory containing MP4 videos to run inference on')

    args = parser.parse_args()
    opt = get_config(args.task)
    opt.mode = "val"
    opt.visual = True
    opt.modelname = args.modelname
    device = torch.device(opt.device)

    # Seed setup for reproducibility
    seed_value = 300
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

    # ========================= Model Loading =========================
    opt.batch_size = args.batch_size * args.n_gpu

    model = get_model(args.modelname, args=args, opt=opt)
    model.to(device)

    checkpoint = torch.load(opt.load_path)
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint.items()}
    model.load_state_dict(new_state_dict)

    criterion = get_criterion(modelname=args.modelname, opt=opt)

    # ========================= If video directory is provided =========================
    if args.input_mp4_directory:
        input_dir = args.input_mp4_directory
        output_dir = input_dir.rstrip('/') + "_OUTPUT"
        os.makedirs(output_dir, exist_ok=True)

        video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
        print(f"Found {len(video_files)} videos in {input_dir}")

        for video_file in video_files:
            video_path = os.path.join(input_dir, video_file)
            
            # Add "_output" suffix before .mp4
            base_name, ext = os.path.splitext(video_file)
            output_video_path = os.path.join(output_dir, f"{base_name}_output{ext}")

            run_inference_on_video(model, device, video_path, output_video_path, args, opt)

        print(f"Processed videos saved to: {output_dir}")
        return


    # ========================= Regular Evaluation Mode =========================
    tf_val = JointTransform2D(
        img_size=args.encoder_input_size,
        low_img_size=args.low_image_size,
        ori_size=opt.img_size,
        crop=opt.crop,
        p_flip=0,
        color_jitter_params=None,
        long_mask=True
    )
    val_dataset = ImageToImage2D(opt.data_path, opt.test_split, tf_val, img_size=args.encoder_input_size, class_id=1)
    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model.eval()
    mean_dice, mean_hdis, mean_iou, mean_acc, mean_se, mean_sp, std_dice, std_hdis, std_iou, std_acc, std_se, std_sp = \
        get_eval(valloader, model, criterion=criterion, opt=opt, args=args)

    print("Evaluation Metrics:")
    print("Mean Dice:        ", mean_dice[1:])
    print("Mean IoU:         ", mean_iou[1:])

if __name__ == '__main__':
    main()
