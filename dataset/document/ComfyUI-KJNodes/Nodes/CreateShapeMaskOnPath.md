---
tags:
- Mask
- MaskGeneration
---

# Create Shape Mask On Path
## Documentation
- Class name: `CreateShapeMaskOnPath`
- Category: `KJNodes/masking/generate`
- Output node: `False`

This node is designed to generate shape-based masks along a specified path, enabling the creation of dynamic and complex mask patterns that can be applied to images or animations. It leverages shape parameters and path coordinates to craft masks that follow a designated trajectory, offering a versatile tool for creative and technical image manipulation tasks.
## Input types
### Required
- **`shape`**
    - Defines the type of shape to be used for the mask, such as circle, square, or triangle. This choice dictates the geometric form of the mask, contributing to its visual style.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`coordinates`**
    - Specifies the center locations for the shape masks along the path. This parameter is crucial for determining the trajectory of the mask creation, impacting the overall design and flow of the generated masks.
    - Comfy dtype: `STRING`
    - Python dtype: `list[str]`
- **`frame_width`**
    - Defines the width of the frame for the mask. This parameter sets the boundary within which the mask is created, influencing the scale and visibility of the shape within the frame.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`frame_height`**
    - Sets the height of the frame for the mask, establishing the vertical boundary and scale of the shape within the generated mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`shape_width`**
    - Determines the width of the shape used in the mask. This parameter affects the size of the shape, allowing for customization of the mask's appearance.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`shape_height`**
    - Specifies the height of the shape used in the mask, influencing the vertical dimension of the shape and thus the mask's overall appearance.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`size_multiplier`**
    - A multiplier that adjusts the size of the shape, offering additional control over the mask's scale beyond the basic shape dimensions.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The primary output, a mask or a batch of masks, created based on the specified shape and path parameters.
    - Python dtype: `torch.Tensor`
- **`mask_inverted`**
    - Comfy dtype: `MASK`
    - An inverted version of the primary mask, providing an alternative masking option where the shape's area is transparent and the surrounding area is opaque.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CreateShapeMaskOnPath:
    
    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask_inverted",)
    FUNCTION = "createshapemask"
    CATEGORY = "KJNodes/masking/generate"
    DESCRIPTION = """
Creates a mask or batch of masks with the specified shape.  
Locations are center locations.  
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
                "coordinates": ("STRING", {"forceInput": True}),
                "frame_width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "frame_height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "shape_width": ("INT", {"default": 128,"min": 8, "max": 4096, "step": 1}),
                "shape_height": ("INT", {"default": 128,"min": 8, "max": 4096, "step": 1}),
        },
        "optional": {
            "size_multiplier": ("FLOAT", {"default": [1.0], "forceInput": True}),
        }
    } 

    def createshapemask(self, coordinates, frame_width, frame_height, shape_width, shape_height, shape, size_multiplier=[1.0]):
        # Define the number of images in the batch
        coordinates = coordinates.replace("'", '"')
        coordinates = json.loads(coordinates)

        batch_size = len(coordinates)
        out = []
        color = "white"
        if len(size_multiplier) != batch_size:
            size_multiplier = size_multiplier * (batch_size // len(size_multiplier)) + size_multiplier[:batch_size % len(size_multiplier)]
        for i, coord in enumerate(coordinates):
            image = Image.new("RGB", (frame_width, frame_height), "black")
            draw = ImageDraw.Draw(image)

            # Calculate the size for this frame and ensure it's not less than 0
            current_width = max(0, shape_width + i * size_multiplier[i])
            current_height = max(0, shape_height + i * size_multiplier[i])

            location_x = coord['x']
            location_y = coord['y']

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
