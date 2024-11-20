# Calculate Upscale
## Documentation
- Class name: `Calculate Upscale`
- Category: `Hakkun`
- Output node: `False`

The Calculate Upscale node is designed to compute the necessary upscale factor and tile size for an image given a target height and the number of tiles in the horizontal direction. It aims to facilitate image processing tasks by determining how much an image needs to be upscaled to meet specific dimensions, thereby assisting in optimizing image manipulation and rendering processes.
## Input types
### Required
- **`image`**
    - The image parameter represents the input image to be processed. It is crucial for determining the current dimensions of the image, which are necessary to calculate the upscale factor and tile size.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`target_height`**
    - The target_height parameter specifies the desired height of the image after upscaling. It plays a key role in calculating the upscale factor by comparing it with the current height of the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`tiles_in_x`**
    - The tiles_in_x parameter indicates the number of horizontal tiles the upscaled image should be divided into. This is essential for calculating the width of each tile after the image has been upscaled.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`tile_size`**
    - Comfy dtype: `INT`
    - The tile_size output represents the width of each tile in the upscaled image. It is calculated based on the upscaled width of the image and the number of tiles in the horizontal direction.
    - Python dtype: `float`
- **`upscale`**
    - Comfy dtype: `FLOAT`
    - The upscale output indicates the factor by which the image needs to be upscaled to achieve the target height. It is a crucial metric for resizing images in image processing tasks.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CalculateUpscale:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IMAGE_TYPE,),
                "target_height": (INT_TYPE, {"default": 1920, "min": 0, "step": 1}),
                "tiles_in_x": (INT_TYPE, {"default": 1, "min": 1, "step": 1}),
            }
        }

    RETURN_TYPES = (INT_TYPE, "FLOAT")
    RETURN_NAMES = ("tile_size","upscale")
    FUNCTION = "calculate"

    CATEGORY = "Hakkun"

    def calculate(self, image, tiles_in_x, target_height):
        image_size = image.size()
        img_width = int(image_size[2])
        img_height = int(image_size[1])

        upscale = target_height/img_height

        upscaled_width = img_width * upscale

        tile_size=upscaled_width/tiles_in_x

        return tile_size, upscale

```
