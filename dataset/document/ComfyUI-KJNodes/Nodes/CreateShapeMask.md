---
tags:
- Mask
- MaskGeneration
---

# Create Shape Mask
## Documentation
- Class name: `CreateShapeMask`
- Category: `KJNodes/masking/generate`
- Output node: `False`

This node is designed to generate a series of masks or a single mask with a specified shape, allowing for dynamic creation of animated masks by adjusting the shape's size over a sequence of frames. It supports customization of the mask's dimensions, shape, and growth per frame, making it versatile for various masking applications.
## Input types
### Required
- **`shape`**
    - Specifies the geometric shape of the mask to be created. It determines the visual form of the mask, affecting its appearance and utility in masking operations.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`frames`**
    - Defines the number of frames (or masks) to generate. This allows for the creation of animated masks by specifying more than one frame.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`location_x`**
    - The x-coordinate of the shape's center location within the mask. It determines where the shape will be positioned horizontally.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`location_y`**
    - The y-coordinate of the shape's center location within the mask. It determines where the shape will be positioned vertically.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`grow`**
    - Specifies the amount by which the shape's size should increase or decrease across frames, enabling dynamic resizing for animated effects.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`frame_width`**
    - The width of the mask frame, defining the horizontal dimension of the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`frame_height`**
    - The height of the mask frame, defining the vertical dimension of the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`shape_width`**
    - The initial width of the shape within the mask, determining its size.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`shape_height`**
    - The initial height of the shape within the mask, determining its size.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The generated mask with the specified shape and dimensions.
    - Python dtype: `numpy.ndarray`
- **`mask_inverted`**
    - Comfy dtype: `MASK`
    - An inverted version of the generated mask, where the shape is transparent and the rest of the mask is opaque.
    - Python dtype: `numpy.ndarray`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [MaskToImage](../../Comfy/Nodes/MaskToImage.md)



## Source code
```python
class CreateShapeMask:
    
    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask_inverted",)
    FUNCTION = "createshapemask"
    CATEGORY = "KJNodes/masking/generate"
    DESCRIPTION = """
Creates a mask or batch of masks with the specified shape.  
Locations are center locations.  
Grow value is the amount to grow the shape on each frame, creating animated masks.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "shape": (
            [   'circle',
                'square',
                'triangle',
            ],
            {
            "default": 'circle'
             }),
                "frames": ("INT", {"default": 1,"min": 1, "max": 4096, "step": 1}),
                "location_x": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
                "location_y": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
                "grow": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "frame_width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "frame_height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "shape_width": ("INT", {"default": 128,"min": 8, "max": 4096, "step": 1}),
                "shape_height": ("INT", {"default": 128,"min": 8, "max": 4096, "step": 1}),
        },
    } 

    def createshapemask(self, frames, frame_width, frame_height, location_x, location_y, shape_width, shape_height, grow, shape):
        # Define the number of images in the batch
        batch_size = frames
        out = []
        color = "white"
        for i in range(batch_size):
            image = Image.new("RGB", (frame_width, frame_height), "black")
            draw = ImageDraw.Draw(image)

            # Calculate the size for this frame and ensure it's not less than 0
            current_width = max(0, shape_width + i*grow)
            current_height = max(0, shape_height + i*grow)

            if shape == 'circle' or shape == 'square':
                # Define the bounding box for the shape
                left_up_point = (location_x - current_width // 2, location_y - current_height // 2)
                right_down_point = (location_x + current_width // 2, location_y + current_height // 2)
                two_points = [left_up_point, right_down_point]

                if shape == 'circle':
                    draw.ellipse(two_points, fill=color)
                elif shape == 'square':
                    draw.rectangle(two_points, fill=color)
                    
            elif shape == 'triangle':
                # Define the points for the triangle
                left_up_point = (location_x - current_width // 2, location_y + current_height // 2) # bottom left
                right_down_point = (location_x + current_width // 2, location_y + current_height // 2) # bottom right
                top_point = (location_x, location_y - current_height // 2) # top point
                draw.polygon([top_point, left_up_point, right_down_point], fill=color)

            image = pil2tensor(image)
            mask = image[:, :, :, 0]
            out.append(mask)
        outstack = torch.cat(out, dim=0)
        return (outstack, 1.0 - outstack,)

```
