---
tags:
- Image
---

# Image Concatenate
## Documentation
- Class name: `ImageConcanate`
- Category: `KJNodes/image`
- Output node: `False`

The ImageConcatenate node is designed for combining two images into a single image along a specified direction. It supports matching the sizes of the input images and adjusting them to have the same batch size if necessary, allowing for flexible image concatenation operations.
## Input types
### Required
- **`image1`**
    - The first image tensor to be concatenated. It plays a crucial role in determining the final image's layout and size, especially when matching image sizes or batch sizes.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image2`**
    - The second image tensor to be concatenated with the first one. Its size and batch size can be adjusted to match the first image, ensuring seamless concatenation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`direction`**
    - Specifies the direction ('right', 'left', 'up', 'down') in which the second image should be concatenated to the first one, influencing the final image's orientation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`match_image_size`**
    - A boolean flag indicating whether the sizes of the two images should be matched before concatenation, affecting the resizing operation if true.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting tensor after concatenating the two input images along the specified direction.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageConcanate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "direction": (
            [   'right',
                'down',
                'left',
                'up',
            ],
            {
            "default": 'right'
             }),
            "match_image_size": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concanate"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Concatenates the image2 to image1 in the specified direction.
"""

    def concanate(self, image1, image2, direction, match_image_size, first_image_shape=None):
        # Check if the batch sizes are different
        batch_size1 = image1.size(0)
        batch_size2 = image2.size(0)

        if batch_size1 != batch_size2:
            # Calculate the number of repetitions needed
            max_batch_size = max(batch_size1, batch_size2)
            repeats1 = max_batch_size // batch_size1
            repeats2 = max_batch_size // batch_size2
            
            # Repeat the images to match the largest batch size
            image1 = image1.repeat(repeats1, 1, 1, 1)
            image2 = image2.repeat(repeats2, 1, 1, 1)
        if match_image_size:
            first_image_shape = first_image_shape if first_image_shape is not None else image1.shape
            image2_resized = image2.movedim(-1,1)
            image2_resized = common_upscale(image2_resized, first_image_shape[2], first_image_shape[1], "lanczos", "disabled").movedim(1,-1)
        else:
            image2_resized = image2
        if direction == 'right':
            row = torch.cat((image1, image2_resized), dim=2)
        elif direction == 'down':
            row = torch.cat((image1, image2_resized), dim=1)
        elif direction == 'left':
            row = torch.cat((image2_resized, image1), dim=2)
        elif direction == 'up':
            row = torch.cat((image2_resized, image1), dim=1)
        return (row,)

```
