# LayerColor: LAB
## Documentation
- Class name: `LayerColor: LAB`
- Category: `😺dzNodes/LayerColor`
- Output node: `False`

The LayerColor: LAB node is designed for color correction in the LAB color space. It adjusts the L, A, and B channels of an image, allowing for precise control over the lightness, green-red, and blue-yellow components, respectively.
## Input types
### Required
- **`image`**
    - The input image to be color corrected. It serves as the base for applying LAB color space adjustments.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`L`**
    - Adjusts the lightness of the image. A positive value increases lightness, while a negative value decreases it.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`A`**
    - Modifies the green-red axis of the image. Positive values shift towards red, negative values towards green.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`B`**
    - Alters the blue-yellow axis of the image. Positive values shift towards yellow, negative values towards blue.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The color-corrected image in the LAB color space, reflecting adjustments made to the L, A, and B channels.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ColorCorrectLAB:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "L": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "A": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "B": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_correct_LAB'
    CATEGORY = '😺dzNodes/LayerColor'

    def color_correct_LAB(self, image, L, A, B):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            __image = tensor2pil(i)
            _l, _a, _b = tensor2pil(i).convert('LAB').split()
            if L != 0 :
                _l = image_gray_offset(_l, L)
            if A != 0 :
                _a = image_gray_offset(_a, A)
            if B != 0 :
                _b = image_gray_offset(_b, B)
            ret_image = image_channel_merge((_l, _a, _b), 'LAB')

            if __image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, __image.split()[-1])

            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
