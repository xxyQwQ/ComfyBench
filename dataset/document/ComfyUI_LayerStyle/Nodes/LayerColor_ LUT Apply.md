---
tags:
- Color
- ImageEnhancement
---

# LayerColor: LUT Apply
## Documentation
- Class name: `LayerColor: LUT Apply`
- Category: `😺dzNodes/LayerColor`
- Output node: `False`

The node applies a Look-Up Table (LUT) to images for color correction, supporting different color spaces. It enables the transformation of the color palette of images according to predefined LUT files, facilitating advanced color grading and adjustments.
## Input types
### Required
- **`image`**
    - The input image(s) to which the LUT will be applied. This is crucial for defining the visual content that will undergo color transformation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`LUT`**
    - Specifies the LUT file to be used for color correction. This determines the specific color transformation that will be applied to the input images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `LUT_LIST`
- **`color_space`**
    - Defines the color space of the input image(s), such as 'linear' or 'log', affecting how the LUT is applied and the final color correction outcome.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image(s) after applying the LUT, showcasing the result of the color correction process.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ColorCorrectLUTapply:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        color_space_list = ['linear', 'log']
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "LUT": (LUT_LIST,),  # LUT文件
                "color_space":  (color_space_list,),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_correct_LUTapply'
    CATEGORY = '😺dzNodes/LayerColor'

    def color_correct_LUTapply(self, image, LUT, color_space):
        ret_images = []
        for i in image:
            i = torch.unsqueeze(i, 0)
            _image = tensor2pil(i)

            lut_file = LUT_DICT[LUT]
            ret_image = apply_lut(_image, lut_file, log=(color_space == 'log'))

            if _image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, _image.split()[-1])
            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
