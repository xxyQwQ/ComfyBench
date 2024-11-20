---
tags:
- Latent
---

# Empty Latent Image Presets
## Documentation
- Class name: `EmptyLatentImagePresets`
- Category: `KJNodes`
- Output node: `False`

This node is designed to generate presets for creating empty latent images with specific dimensions, offering options for inversion and batch processing. It abstracts the complexity of generating latent images by providing predefined dimension options and handling the inversion logic internally, making it easier to produce latent images tailored to specific requirements.
## Input types
### Required
- **`dimensions`**
    - Specifies the dimensions of the latent image to be generated. The selection of dimensions directly influences the size of the output latent image, with several predefined options available.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`invert`**
    - Determines whether the dimensions of the latent image should be inverted. This option allows for flexibility in the orientation of the generated latent image.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`batch_size`**
    - Defines the number of latent images to generate in a single batch. This allows for efficient generation of multiple latent images at once.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`Latent`**
    - Comfy dtype: `LATENT`
    - The generated latent image(s) based on the specified dimensions and inversion setting.
    - Python dtype: `torch.Tensor`
- **`Width`**
    - Comfy dtype: `INT`
    - The width of the generated latent image.
    - Python dtype: `int`
- **`Height`**
    - Comfy dtype: `INT`
    - The height of the generated latent image.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [Anything Everywhere](../../cg-use-everywhere/Nodes/Anything Everywhere.md)



## Source code
```python
class EmptyLatentImagePresets:
    @classmethod
    def INPUT_TYPES(cls):  
        return {
        "required": {
            "dimensions": (
            [   '512 x 512',
                '768 x 512',
                '960 x 512',
                '1024 x 512',
                '1536 x 640',
                '1344 x 768',
                '1216 x 832',
                '1152 x 896',
                '1024 x 1024',
            ],
            {
            "default": '512 x 512'
             }),
           
            "invert": ("BOOLEAN", {"default": False}),
            "batch_size": ("INT", {
            "default": 1,
            "min": 1,
            "max": 4096
            }),
        },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("Latent", "Width", "Height")
    FUNCTION = "generate"
    CATEGORY = "KJNodes"

    def generate(self, dimensions, invert, batch_size):
        from nodes import EmptyLatentImage
        result = [x.strip() for x in dimensions.split('x')]
        
        if invert:
            width = int(result[1].split(' ')[0])
            height = int(result[0])
        else:
            width = int(result[0])
            height = int(result[1].split(' ')[0])
        latent = EmptyLatentImage().generate(width, height, batch_size)[0]

        return (latent, int(width), int(height),)

```
