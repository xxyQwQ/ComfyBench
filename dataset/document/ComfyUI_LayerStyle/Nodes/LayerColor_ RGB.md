# LayerColor: RGB
## Documentation
- Class name: `LayerColor: RGB`
- Category: `😺dzNodes/LayerColor`
- Output node: `False`

This node is designed for color correction in RGB color space, allowing adjustments to the red, green, and blue channels of an image. It enables fine-tuning of the image's color balance by applying specific offsets to each color channel, enhancing or modifying the image's overall coloration.
## Input types
### Required
- **`image`**
    - The input image to be color corrected. It serves as the base for applying RGB adjustments, impacting the visual outcome of the color correction process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`R`**
    - The offset value for the red channel, which adjusts the intensity of red colors in the image. This parameter directly influences the image's red hues, allowing for nuanced color correction.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`G`**
    - The offset value for the green channel, which adjusts the intensity of green colors in the image. This parameter is crucial for modifying the image's green hues, enabling detailed color adjustments.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`B`**
    - The offset value for the blue channel, which adjusts the intensity of blue colors in the image. By altering this parameter, users can fine-tune the blue hues within the image, achieving desired color effects.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The color-corrected image, with adjustments applied to the red, green, and blue channels. This output reflects the cumulative effect of the RGB offsets on the original image, showcasing the enhanced or altered coloration.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ColorCorrectRGB:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "R": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "G": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "B": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_correct_RGB'
    CATEGORY = '😺dzNodes/LayerColor'

    def color_correct_RGB(self, image, R, G, B):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i,0)
            __image = tensor2pil(i)
            _r, _g, _b = tensor2pil(i).convert('RGB').split()
            if R != 0 :
                _r = image_gray_offset(_r, R)
            if G != 0 :
                _g = image_gray_offset(_g, G)
            if B != 0 :
                _b = image_gray_offset(_b, B)
            ret_image = image_channel_merge((_r, _g, _b), 'RGB')

            if __image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, __image.split()[-1])

            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
