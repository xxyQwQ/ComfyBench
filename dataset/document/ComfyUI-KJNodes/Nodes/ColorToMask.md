---
tags:
- ImageMask
- Mask
- MaskGeneration
---

# Color To Mask
## Documentation
- Class name: `ColorToMask`
- Category: `KJNodes/masking`
- Output node: `False`

The `ColorToMask` node is designed to convert specific RGB values within images into a binary mask, allowing for selective processing or analysis of those areas. It supports batch processing to handle multiple images simultaneously, offering customization through parameters like color selection, inversion, and thresholding for flexible mask generation.
## Input types
### Required
- **`images`**
    - Specifies the images to be processed. This parameter is crucial for defining the input data on which the mask will be applied.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`invert`**
    - Determines whether the mask should be inverted. Inverting the mask swaps the areas to be masked and preserved, affecting the final output.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`red`**
    - Specifies the red component of the RGB value to be masked. This parameter is part of the color selection for mask generation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`green`**
    - Specifies the green component of the RGB value to be masked. Along with red and blue, it defines the specific color to target for masking.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`blue`**
    - Specifies the blue component of the RGB value to be masked. It's a critical part of defining the exact color that will be converted into a mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`threshold`**
    - Defines the tolerance for color matching. A higher threshold allows for a broader range of similar colors to be included in the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`per_batch`**
    - Controls the number of images processed in each batch. This parameter helps manage memory usage and processing speed when handling multiple images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output is a binary mask where the specified RGB color within the images has been converted to white (or black if inverted), and all other areas are the opposite color.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ColorToMask:
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "clip"
    CATEGORY = "KJNodes/masking"
    DESCRIPTION = """
Converts chosen RGB value to a mask.  
With batch inputs, the **per_batch**  
controls the number of images processed at once.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
                 "invert": ("BOOLEAN", {"default": False}),
                 "red": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
                 "green": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
                 "blue": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
                 "threshold": ("INT", {"default": 10,"min": 0, "max": 255, "step": 1}),
                 "per_batch": ("INT", {"default": 16, "min": 1, "max": 4096, "step": 1}),
        },
    } 

    def clip(self, images, red, green, blue, threshold, invert, per_batch):

        color = torch.tensor([red, green, blue], dtype=torch.uint8)  
        black = torch.tensor([0, 0, 0], dtype=torch.uint8)
        white = torch.tensor([255, 255, 255], dtype=torch.uint8)
        
        if invert:
            black, white = white, black

        steps = images.shape[0]
        pbar = ProgressBar(steps)
        tensors_out = []
        
        for start_idx in range(0, images.shape[0], per_batch):

            # Calculate color distances
            color_distances = torch.norm(images[start_idx:start_idx+per_batch] * 255 - color, dim=-1)
            
            # Create a mask based on the threshold
            mask = color_distances <= threshold
            
            # Apply the mask to create new images
            mask_out = torch.where(mask.unsqueeze(-1), white, black).float()
            mask_out = mask_out.mean(dim=-1)

            tensors_out.append(mask_out.cpu())
            batch_count = mask_out.shape[0]
            pbar.update(batch_count)
       
        tensors_out = torch.cat(tensors_out, dim=0)
        tensors_out = torch.clamp(tensors_out, min=0.0, max=1.0)
        return tensors_out,

```
