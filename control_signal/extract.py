from .pose import SkeletonGeneratorAlign
from .strand_mapper import StrandMapper
import numpy as np
import cv2
from tqdm import tqdm
import os
import subprocess
from PIL import Image
import time
import torch
import torch.nn.functional as F

def convert_to_h264(input_path, crf=23):
    """
    Convert video to H.264 using ffmpeg and replace the original file

    Args:
        input_path (str): Path to the input video file
        crf (int): Constant Rate Factor for quality (lower = better quality)
    """
    try:
        # Create temporary output path
        input_path = str(input_path)
        temp_path = input_path.replace('.mp4', '_temp_h264.mp4')

        # Run ffmpeg command
        cmd = [
            'ffmpeg', '-i', input_path,
            '-vcodec', 'libx264',
            '-crf', str(crf),
            '-y',  # Overwrite output file if it exists
            temp_path
        ]

        # Execute ffmpeg command with suppressed output
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # Replace original file with converted one
            os.replace(temp_path, input_path)
            print(f"Converted to H.264: {input_path}")
        else:
            print(f"Warning: Failed to convert {input_path} to H.264")
            print(f"FFmpeg error: {result.stderr}")
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except FileNotFoundError:
        print("Warning: ffmpeg not found. Install ffmpeg to enable H.264 conversion.")
    except Exception as e:
        print(f"Warning: Error converting {input_path} to H.264: {e}")
        # Clean up temp file if it exists
        temp_path = input_path.replace('.mp4', '_temp_h264.mp4')
        if os.path.exists(temp_path):
            os.remove(temp_path)


def get_target_from_src(face_align_params, hair_mask):
    """Placeholder function - implement as needed"""
    pass


def get_target_to_src_mapping(face_align_params, source_image):
    """
    Perform target-to-source mapping using face alignment parameters.

    Args:
        face_align_params (dict): Face alignment parameters containing transformation info
        source_image (np.ndarray): Source image tensor [H, W, C]

    Returns:
        np.ndarray: Target image after sampling from source [H, W, C]
    """
    # Get target shape from hair mask
    target_h, target_w = source_image.shape[0:2]
    source_image = torch.from_numpy(source_image).permute(2, 0, 1).unsqueeze(0).float().cuda()
    device = source_image.device

    #
    # import ipdb; ipdb.set_trace()

    # Extract alignment parameters and convert to torch tensors
    axis_ratio = torch.tensor(face_align_params['axis_ratio'], device=device, dtype=torch.float32)
    ortho_ratio = torch.tensor(face_align_params['ortho_ratio'], device=device, dtype=torch.float32)
    target_center = torch.tensor(face_align_params['target_center'], device=device, dtype=torch.float32)
    orig_center = torch.tensor(face_align_params['orig_center'], device=device, dtype=torch.float32)
    orig_axis_unit = torch.tensor(face_align_params['orig_axis_unit'], device=device, dtype=torch.float32)
    orig_ortho_unit = torch.tensor(face_align_params['orig_ortho_unit'], device=device, dtype=torch.float32)
    target_axis_unit = torch.tensor(face_align_params['target_axis_unit'], device=device, dtype=torch.float32)
    target_ortho_unit = torch.tensor(face_align_params['target_ortho_unit'], device=device, dtype=torch.float32)

    # Generate target coordinate grid (0-1 normalized space)
    y_coords = torch.linspace(0, 1, target_h, device=device, dtype=torch.float32)
    x_coords = torch.linspace(0, 1, target_w, device=device, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Stack to get coordinate grid [H, W, 2]
    target_coords = torch.stack([x_grid, y_grid], dim=-1)  # [H, W, 2] (x, y)

    # Reshape for batch processing [H*W, 2]
    target_coords_flat = target_coords.view(-1, 2)

    # Step 1: Convert target coordinates relative to target center
    target_rel = target_coords_flat - target_center.unsqueeze(0)  # [H*W, 2]

    # Step 2: Project onto target axis and orthogonal axis
    target_axis_proj = torch.sum(target_rel * target_axis_unit.unsqueeze(0), dim=1)  # [H*W]
    target_ortho_proj = torch.sum(target_rel * target_ortho_unit.unsqueeze(0), dim=1)  # [H*W]

    # Step 3: Apply inverse scaling (1/ratio to go back to original scale)
    orig_axis_proj = target_axis_proj / axis_ratio
    orig_ortho_proj = target_ortho_proj / ortho_ratio

    # Step 4: Reconstruct in original space using original axis vectors
    orig_coords = (orig_axis_proj.unsqueeze(1) * orig_axis_unit.unsqueeze(0) +
                   orig_ortho_proj.unsqueeze(1) * orig_ortho_unit.unsqueeze(0))  # [H*W, 2]

    # Step 5: Add original center to get back to origin space (0-1)
    orig_coords = orig_coords + orig_center.unsqueeze(0)  # [H*W, 2]

    # Step 6: Convert 0-1 coordinates to pixel coordinates for source image
    source_h, source_w = source_image.shape[-2:]
    pixel_coords = orig_coords.clone()
    pixel_coords[:, 0] = pixel_coords[:, 0] * source_w  # x coordinates
    pixel_coords[:, 1] = pixel_coords[:, 1] * source_h  # y coordinates

    # Step 7: Normalize coordinates for grid_sample (range [-1, 1])
    grid_coords = pixel_coords.clone()
    grid_coords[:, 0] = (grid_coords[:, 0] / source_w) * 2.0 - 1.0  # x: [0, W] -> [-1, 1]
    grid_coords[:, 1] = (grid_coords[:, 1] / source_h) * 2.0 - 1.0  # y: [0, H] -> [-1, 1]

    # Reshape grid for grid_sample [1, H, W, 2]
    grid = grid_coords.view(1, target_h, target_w, 2)

    # Step 8: Perform bilinear sampling
    # Add batch dimension to source image if needed
    if source_image.dim() == 3:
        source_image = source_image.unsqueeze(0)  # [1, C, H, W]
    # import ipdb; ipdb.set_trace

    # Sample from source image
    sampled = F.grid_sample(
        source_image,
        grid,
        mode='bilinear',
        padding_mode='zeros',  # Use border padding for out-of-bounds pixels
        align_corners=True
    )


    # Remove batch dimension and return [C, H, W]
    return sampled.squeeze(0).cpu().numpy().transpose(1, 2, 0)


class WrappedPreprocessorAlign:
    def __init__(self, drop_head=False, drop_face=False, drop_hair=False, disable_align=False):
        self.skeleton_generator = SkeletonGeneratorAlign(drop_head=drop_head, drop_face=drop_face)
        self.strand_mapper = StrandMapper()
        self.drop_hair = drop_hair
        self.disable_align = disable_align
        self.use_align = not disable_align

    def __call__(self, image, ref_image, first_frame=False, previous_params=None, hair_seg=None, video_frame_wo_hair=None):
        times = {}
        total_start = time.perf_counter()

        # Step 1: Generate skeleton
        step_start = time.perf_counter()
        if video_frame_wo_hair is not None:
            skeleton, face_align_params, _, _ = self.skeleton_generator(ref_image, video_frame_wo_hair, first_frame, previous_params, use_align=self.use_align)
        else:
            skeleton, face_align_params, _, _ = self.skeleton_generator(ref_image, image, first_frame, previous_params, use_align=self.use_align)
        times['skeleton_generation'] = time.perf_counter() - step_start

        # Step 2: Color conversion BGR to RGB
        step_start = time.perf_counter()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        times['color_conversion'] = time.perf_counter() - step_start

        # Step 3: Convert to PIL image
        step_start = time.perf_counter()
        image_pil = Image.fromarray(image)
        times['pil_conversion'] = time.perf_counter() - step_start

        # Step 4: Generate hair mask
        if not self.drop_hair:
            step_start = time.perf_counter()
            if hair_seg is not None:
                # convert to 0-1
                hair_mask = hair_seg / 255.0
                hair_mask = hair_mask[:, :, 0]
                # import ipdb; ipdb.set_trace()
            else:
                raise ValueError("hair_seg is required for simulation control extraction")
            times['hair_mask_generation'] = time.perf_counter() - step_start

            # Step 5: Generate orientation map
            step_start = time.perf_counter()
            orientation = self.strand_mapper(image, hair_mask)
            times['orientation_mapping'] = time.perf_counter() - step_start

            # Step 6: Resize operations
            step_start = time.perf_counter()
            h, w = skeleton.shape[:2]
            if orientation.shape[:2] != (h, w):
                orientation = cv2.resize(orientation, (w, h), interpolation=cv2.INTER_LINEAR)
            if hair_mask.shape[:2] != (h, w):
                hair_mask = cv2.resize(hair_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            times['resize_operations'] = time.perf_counter() - step_start

            # Step 7: Alpha blending
            # import ipdb; ipdb.set_trace()
            hair_mask = hair_mask[:,:,np.newaxis]

            # if self.use_align:
            hair_mask = get_target_to_src_mapping(face_align_params, hair_mask)
            orientation = get_target_to_src_mapping(face_align_params, orientation)
            # else:
            #     hair_mask = hair_mask
            #     orientation = orientation

            step_start = time.perf_counter()
            alpha = hair_mask
            if skeleton.ndim == 3 and alpha.ndim == 2:
                alpha = alpha[..., None]

            blended = (orientation.astype(np.float32) * alpha +
                    skeleton.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
            times['alpha_blending'] = time.perf_counter() - step_start

            total_time = time.perf_counter() - total_start
        else:
            blended = skeleton
            total_time = time.perf_counter() - total_start

        # Print timing analysis
        # print("\n=== Timing Analysis ===")
        # print(f"1. Skeleton Generation:    {times['skeleton_generation']*1000:.2f} ms ({times['skeleton_generation']/total_time*100:.1f}%)")
        # print(f"2. Color Conversion:       {times['color_conversion']*1000:.2f} ms ({times['color_conversion']/total_time*100:.1f}%)")
        # print(f"3. PIL Conversion:         {times['pil_conversion']*1000:.2f} ms ({times['pil_conversion']/total_time*100:.1f}%)")
        # print(f"4. Hair Mask Generation:   {times['hair_mask_generation']*1000:.2f} ms ({times['hair_mask_generation']/total_time*100:.1f}%)")
        # print(f"5. Orientation Mapping:    {times['orientation_mapping']*1000:.2f} ms ({times['orientation_mapping']/total_time*100:.1f}%)")
        # print(f"6. Resize Operations:      {times['resize_operations']*1000:.2f} ms ({times['resize_operations']/total_time*100:.1f}%)")
        # print(f"7. Alpha Blending:         {times['alpha_blending']*1000:.2f} ms ({times['alpha_blending']/total_time*100:.1f}%)")
        # print(f"Total Processing Time:     {total_time*1000:.2f} ms")
        # print("=====================\n")

        return blended, face_align_params


class WrappedVideoPreprocessorAlign:
    def __init__(self, drop_head=False, drop_face=False, drop_hair=False, disable_align=False):
        self.wrapped_preprocessor = WrappedPreprocessorAlign(drop_head=drop_head, drop_face=drop_face, drop_hair=drop_hair, disable_align=disable_align)


    def __call__(self, video_path, ref_image_path, output_path, hair_seg_video_path=None, video_path_wo_hair=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video_path}")

        ref_image = cv2.imread(ref_image_path)

        if hair_seg_video_path is not None:
            hair_seg_cap = cv2.VideoCapture(hair_seg_video_path)
            if not hair_seg_cap.isOpened():
                raise RuntimeError(f"Could not open hair segmentation video {hair_seg_video_path}")

        if video_path_wo_hair is not None:
            video_cap_wo_hair = cv2.VideoCapture(video_path_wo_hair)
            if not video_cap_wo_hair.isOpened():
                raise RuntimeError(f"Could not open video {video_path_wo_hair}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, frame = cap.read()
        hair_seg_frame = None
        video_frame_wo_hair = None
        if hair_seg_video_path is not None:
            ret_hair_seg, hair_seg_frame = hair_seg_cap.read()
        if video_path_wo_hair is not None:
            ret_video_wo_hair, video_frame_wo_hair = video_cap_wo_hair.read()
        if not ret:
            cap.release()
            raise RuntimeError(f"Could not read first frame from {video_path}")




        blended_first, face_align_params = self.wrapped_preprocessor(frame, ref_image, first_frame=True, previous_params=None, hair_seg=hair_seg_frame, video_frame_wo_hair=video_frame_wo_hair)
        if blended_first.ndim == 2:
            blended_first = cv2.cvtColor(blended_first, cv2.COLOR_GRAY2BGR)

        h, w = blended_first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not out.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open video writer for {output_path}")
        out.write(blended_first)

        for _ in range(frame_count - 1):
            ret, frame = cap.read()
            hair_seg_frame = None
            video_frame_wo_hair = None
            if hair_seg_video_path is not None:
                ret_hair_seg, hair_seg_frame = hair_seg_cap.read()
            if video_path_wo_hair is not None:
                ret_video_wo_hair, video_frame_wo_hair = video_cap_wo_hair.read()
            if not ret:
                break

            blended, face_align_params = self.wrapped_preprocessor(frame, ref_image, first_frame=False, previous_params=face_align_params, hair_seg=hair_seg_frame, video_frame_wo_hair=video_frame_wo_hair)
            if blended.ndim == 2:
                blended = cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)
            out.write(blended)

        cap.release()
        out.release()
        # import ipdb; ipdb.set_trace()
        # cv2.destroyAllWindows()
        # import ipdb; ipdb.set_trace()
        print(f"output_path: {output_path}")
        convert_to_h264(output_path)
        print(f"\nVideo processing complete. Output saved to {output_path}")
