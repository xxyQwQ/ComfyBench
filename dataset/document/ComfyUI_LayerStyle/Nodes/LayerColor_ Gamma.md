# LayerColor: Gamma
## Documentation
- Class name: `LayerColor: Gamma`
- Category: `😺dzNodes/LayerColor`
- Output node: `False`

The Gamma node is designed for color correction by applying gamma correction to images. It adjusts the luminance of the images based on the gamma value provided, enhancing the visual quality of images by making them appear more natural or stylized according to the gamma setting.
## Input types
### Required
- **`image`**
    - The input image(s) to which gamma correction will be applied. This parameter is crucial for defining the visual content that will undergo the adjustment process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`gamma`**
    - The gamma value determines the intensity of the luminance adjustment. A higher gamma value darkens the image, while a lower value brightens it, playing a key role in achieving the desired visual effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image(s) after applying gamma correction, showcasing the adjusted luminance levels.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ColorCorrectGamma:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),
                "gamma": ("FLOAT", {"default": 1, "min": 0.1, "max": 10, "step": 0.01}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_correct_gamma'
    CATEGORY = '😺dzNodes/LayerColor'

    def color_correct_gamma(self, image, gamma):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            __image = tensor2pil(i)
            ret_image = gamma_trans(tensor2pil(i), gamma)

            if __image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, __image.split()[-1])

            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
