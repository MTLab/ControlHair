import numpy as np
import cv2
import os
import sys
import torch
import torch.nn.functional as F
import imageio
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HAIRSTEP_ROOT = Path(
    os.environ.get("CONTROLHAIR_HAIRSTEP_ROOT", REPO_ROOT / "third_party" / "HairStep")
).expanduser().resolve()
if not (HAIRSTEP_ROOT / "lib").is_dir():
    raise RuntimeError(
        "HairStep is not prepared. Run scripts/prepare_external_dependencies.py "
        "--component hairstep --accept-restricted-licenses"
    )
sys.path.insert(0, str(HAIRSTEP_ROOT))
from lib.model.img2hairstep.UNet import Model

class StrandMapper:
    def __init__(self, model_path=None):
        """
        Initialize the StrandMapper with the strand map model.

        Args:
            model_path (str): Path to the strand map model checkpoint
        """
        if model_path is None:
            model_path = HAIRSTEP_ROOT / "checkpoints" / "img2hairstep" / "img2strand.pth"
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"HairStep strand checkpoint not found: {model_path}. "
                "See the optional physics section in README.md."
            )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.strand_map_model = Model().to(self.device)
        self.strand_map_model.load_state_dict(torch.load(model_path, weights_only=False))
        self.strand_map_model.eval()

    def __call__(self, rgb_image, hair_mask):
        """
        Generate strand map from RGB image and hair mask.

        Args:
            rgb_image (np.ndarray): RGB image (H, W, 3)
            hair_mask (np.ndarray): Hair mask (H, W) with values 0-1

        Returns:
            np.ndarray: Strand map (H, W, 3) with values 0-255
        """
        # Convert inputs to GPU tensors early
        rgb_tensor = torch.from_numpy(rgb_image).float().to(self.device) / 255.0  # Normalize to [0,1]
        mask_tensor = torch.from_numpy(hair_mask).float().to(self.device)

        # Get bbox from hair_mask using GPU operations
        hair_coords = torch.nonzero(mask_tensor > 0, as_tuple=True)

        if len(hair_coords[0]) > 0:
            y_min, y_max = hair_coords[0].min().item(), hair_coords[0].max().item()
            x_min, x_max = hair_coords[1].min().item(), hair_coords[1].max().item()

            # Crop the image and mask using GPU tensors
            cropped_image = rgb_tensor[y_min:y_max+1, x_min:x_max+1]
            cropped_mask = mask_tensor[y_min:y_max+1, x_min:x_max+1]

            # Apply mask to cropped image on GPU
            cropped_image = cropped_image * cropped_mask.unsqueeze(-1)

            # Pad to square using torch.nn.functional.pad
            bbox_h, bbox_w = cropped_image.shape[:2]

            if bbox_h > bbox_w:
                # Need to pad width
                diff = bbox_h - bbox_w
                pad_left = diff // 2
                pad_right = diff - pad_left
                # pad format: (left, right, top, bottom, front, back)
                cropped_image = F.pad(cropped_image.permute(2, 0, 1),
                                    (pad_left, pad_right, 0, 0),
                                    mode='constant', value=0).permute(1, 2, 0)
            elif bbox_w > bbox_h:
                # Need to pad height
                diff = bbox_w - bbox_h
                pad_top = diff // 2
                pad_bottom = diff - pad_top
                cropped_image = F.pad(cropped_image.permute(2, 0, 1),
                                    (0, 0, pad_top, pad_bottom),
                                    mode='constant', value=0).permute(1, 2, 0)
        else:
            # Fallback: use entire image if no hair detected
            cropped_image = rgb_tensor

        # Prepare input tensor for model (already on GPU and normalized)
        input_tensor = cropped_image.permute(2, 0, 1).unsqueeze(0)
        input_tensor = F.interpolate(input_tensor, size=(512, 512),
                                   mode='bilinear', align_corners=False)

        # Generate strand map using the model (all on GPU)
        with torch.no_grad():
            orientation = self.strand_map_model(input_tensor).squeeze(0).permute(1, 2, 0)

        # Post-process the model output on GPU
        orientation = torch.clamp(orientation, 0, 1)
        # Rearrange channels: [G, R, B] where B=1 (constant blue channel)
        orientation = torch.cat([orientation[:,:,1:2], orientation[:,:,0:1],
                               torch.ones((orientation.shape[0], orientation.shape[1], 1),
                                        device=self.device)], dim=2)

        # Convert to 0-255 range on GPU
        orientation = orientation * 255
        orientation = torch.clamp(orientation, 0, 255)

        # Create orientation map with the same shape as the original image
        orientation_map = torch.zeros((rgb_image.shape[0], rgb_image.shape[1], 3),
                                    dtype=torch.float32, device=self.device)

        if len(hair_coords[0]) > 0:
            # Resize back to the padded square size using GPU interpolation
            bbox_h, bbox_w = cropped_image.shape[:2]
            orientation_resized = F.interpolate(orientation.permute(2, 0, 1).unsqueeze(0),
                                              size=(bbox_h, bbox_w),
                                              mode='bilinear', align_corners=False)
            orientation_resized = orientation_resized.squeeze(0).permute(1, 2, 0)

            # Extract the valid (non-padded) region
            orig_bbox_h = y_max - y_min + 1
            orig_bbox_w = x_max - x_min + 1

            if orig_bbox_h > orig_bbox_w:
                # Width was padded, extract the center part
                diff = orig_bbox_h - orig_bbox_w
                pad_left = diff // 2
                valid_orientation = orientation_resized[:, pad_left:pad_left+orig_bbox_w]
            elif orig_bbox_w > orig_bbox_h:
                # Height was padded, extract the center part
                diff = orig_bbox_w - orig_bbox_h
                pad_top = diff // 2
                valid_orientation = orientation_resized[pad_top:pad_top+orig_bbox_h, :]
            else:
                # No padding needed, use as is
                valid_orientation = orientation_resized

            # Place back in the original position
            orientation_map[y_min:y_max+1, x_min:x_max+1] = valid_orientation
        else:
            # If no hair detected, resize the processed image to match original dimensions
            orientation_map = F.interpolate(orientation.permute(2, 0, 1).unsqueeze(0),
                                          size=(rgb_image.shape[0], rgb_image.shape[1]),
                                          mode='bilinear', align_corners=False)
            orientation_map = orientation_map.squeeze(0).permute(1, 2, 0)

        # Convert back to numpy only at the very end
        return orientation_map.cpu().numpy().astype(np.uint8)
