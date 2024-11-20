# Image Pad For Outpaint Target Size
## Documentation
- Class name: `ImagePadForOutpaintTargetSize`
- Category: `image`
- Output node: `False`

This node is designed to adjust the dimensions of an image to meet a specified target size for outpainting purposes. It scales the image while maintaining its aspect ratio, calculates the necessary padding to reach the target dimensions, and optionally applies feathering to blend the edges smoothly. The node can also handle an associated mask, scaling and padding it in tandem with the image.
## Input types
### Required
- **`image`**
    - The input image to be processed for outpainting, which will be scaled and padded to meet the target dimensions.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`target_width`**
    - The desired width of the output image after padding. This parameter determines how much the input image needs to be scaled and padded horizontally.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`target_height`**
    - The desired height of the output image after padding. It influences the vertical scaling and padding required to achieve the target dimensions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`feathering`**
    - Specifies the intensity of the edge feathering applied to the padded areas, enhancing the visual integration of the padded regions with the original image.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`upscale_method`**
    - The method used for upscaling the image during the scaling process, affecting the quality of the resized image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`mask`**
    - An optional mask that, if provided, is scaled and padded in parallel with the image, maintaining the alignment between the image and its mask.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The scaled and padded image, adjusted to match the target dimensions while preserving the original aspect ratio.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The scaled and padded mask corresponding to the processed image, if a mask was provided.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImagePadForOutpaintTargetSize:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "feathering": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "upscale_method": (s.upscale_methods,),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"

    CATEGORY = "image"

    def expand_image(self, image, target_width, target_height, feathering, upscale_method, mask=None):
        B, H, W, C = image.size()
        new_height = 0
        new_width = 0
         # Calculate the scaling factor while maintaining aspect ratio
        scaling_factor = min(target_width / W, target_height / H)
        
        # Check if the image needs to be downscaled
        if scaling_factor < 1:
            image = image.movedim(-1,1)
            # Calculate the new width and height after downscaling
            new_width = int(W * scaling_factor)
            new_height = int(H * scaling_factor)
            
            # Downscale the image
            image_scaled = common_upscale(image, new_width, new_height, upscale_method, "disabled").movedim(1,-1)
            if mask is not None:
                mask_scaled = mask.unsqueeze(0)  # Add an extra dimension for batch size
                mask_scaled = F.interpolate(mask_scaled, size=(new_height, new_width), mode="nearest")
                mask_scaled = mask_scaled.squeeze(0)  # Remove the extra dimension after interpolation
            else:
                mask_scaled = mask
        else:
            # If downscaling is not needed, use the original image dimensions
            image_scaled = image
            mask_scaled = mask

        # Calculate how much padding is needed to reach the target dimensions
        pad_top = max(0, (target_height - new_height) // 2)
        pad_bottom = max(0, target_height - new_height - pad_top)
        pad_left = max(0, (target_width - new_width) // 2)
        pad_right = max(0, target_width - new_width - pad_left)

        # Now call the original expand_image with the calculated padding
        return ImagePadForOutpaintMasked.expand_image(self, image_scaled, pad_left, pad_top, pad_right, pad_bottom, feathering, mask_scaled)

```
