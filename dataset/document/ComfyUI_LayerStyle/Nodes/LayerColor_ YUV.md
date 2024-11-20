# LayerColor: YUV
## Documentation
- Class name: `LayerColor: YUV`
- Category: `😺dzNodes/LayerColor`
- Output node: `False`

The node 'LayerColor: YUV' is designed for color correction in the YUV color space. It adjusts the Y (luminance), U (chrominance), and V (chrominance) components of an image to achieve desired color effects or corrections.
## Input types
### Required
- **`image`**
    - The 'image' parameter represents the input image(s) to be processed. It is crucial for the color correction operation, serving as the base upon which YUV adjustments are applied.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`Y`**
    - The 'Y' parameter adjusts the luminance component of the image. Its modification affects the brightness and contrast of the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`U`**
    - The 'U' parameter modifies the chrominance component related to blue color projection. Altering this value impacts the blue-yellow balance of the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`V`**
    - The 'V' parameter adjusts the chrominance component associated with red color projection, influencing the red-green balance of the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output 'image' is the result of applying YUV color corrections to the input image(s), reflecting the adjustments made to luminance and chrominance components.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ColorCorrectYUV:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "Y": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "U": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "V": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_correct_YUV'
    CATEGORY = '😺dzNodes/LayerColor'

    def color_correct_YUV(self, image, Y, U, V):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            __image = tensor2pil(i)
            _y, _u, _v = tensor2pil(i).convert('YCbCr').split()
            if Y != 0 :
                _y = image_gray_offset(_y, Y)
            if U != 0 :
                _u = image_gray_offset(_u, U)
            if V != 0 :
                _v = image_gray_offset(_v, V)
            ret_image = image_channel_merge((_y, _u, _v), 'YCbCr')

            if __image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, __image.split()[-1])

            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
