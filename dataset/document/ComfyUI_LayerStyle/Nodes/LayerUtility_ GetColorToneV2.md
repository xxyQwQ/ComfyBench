---
tags:
- Color
---

# LayerUtility: GetColorTone V2
## Documentation
- Class name: `LayerUtility: GetColorToneV2`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

The GetColorToneV2 node is designed to analyze and manipulate the color tone of images. It offers advanced features for removing backgrounds, adjusting masks, and determining the main or average color tone of specific image areas, such as the entire image, the background, or the subject. This node is an enhancement over its predecessor, providing more control over the image processing and color analysis.
## Input types
### Required
- **`image`**
    - The input image to be processed for color tone analysis. This parameter is crucial as it serves as the base for all subsequent operations within the node.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mode`**
    - Specifies the method of color tone analysis: either determining the main color or calculating the average color of the image. This choice affects the algorithm used for color extraction.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`color_of`**
    - Determines the area of the image to analyze: the entire image, just the background, or only the subject. This parameter allows for targeted color analysis based on the specified area.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`remove_bkgd_method`**
    - The method used for background removal from the image, offering options like 'BiRefNet' or 'RMBG 1.4'. This is essential for isolating the subject or background for accurate color analysis.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`invert_mask`**
    - A boolean flag to invert the mask applied during image processing, which can alter the area of interest for color analysis.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`mask_grow`**
    - Specifies the amount to grow or shrink the mask around the subject or background, allowing for finer control over the area being analyzed.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`mask`**
    - An optional mask that can be applied to the image for more precise control over the areas to be analyzed for color tone.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The processed image with applied color tone adjustments or background removal effects.
    - Python dtype: `torch.Tensor`
- **`color_in_hex`**
    - Comfy dtype: `STRING`
    - The determined color of the specified image area in HEX format, providing a precise color representation.
    - Python dtype: `str`
- **`HSV color in list`**
    - Comfy dtype: `LIST`
    - The HSV (Hue, Saturation, Value) representation of the determined color, offering an alternative color space for analysis.
    - Python dtype: `list`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class GetColorToneV2:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        remove_background_list = ['BiRefNet', 'RMBG 1.4',]
        subject_list = ['entire', 'background', 'subject']
        mode_list = ['main_color', 'average']
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "mode": (mode_list,),  # 主色/平均色
                "color_of": (subject_list,),
                "remove_bkgd_method": (remove_background_list,),
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask#
                "mask_grow": ("INT", {"default": 16, "min": -999, "max": 999, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "LIST",)
    RETURN_NAMES = ("image", "color_in_hex", "HSV color in list",)
    FUNCTION = 'get_color_tone_v2'
    CATEGORY = '😺dzNodes/LayerUtility'

    def get_color_tone_v2(self, image, mode, remove_bkgd_method, color_of, invert_mask, mask_grow,
                  mask=None
                  ):

        _images = []
        _masks = []
        ret_images = []
        need_rmbg = False
        for i in image:
            _images.append(torch.unsqueeze(i, 0))
            m = tensor2pil(i)
            if m.mode == 'RGBA':
                _masks.append(1 - image2mask(m.split()[-1]))
            else:
                need_rmbg = True
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            _masks = []
            for m in mask:
                if not invert_mask:
                    m = 1 - m
                _masks.append(torch.unsqueeze(m, 0))
            need_rmbg = False

        max_batch = max(len(_images), len(_masks))

        if remove_bkgd_method == 'BiRefNet':
            from .birefnet_func import BiRefNetRemoveBackground
            birefnetrmbg = BiRefNetRemoveBackground()

        for i in range(max_batch):
            _image = _images[i] if i < len(_images) else _images[-1]
            _image = tensor2pil(_image).convert("RGB")
            if need_rmbg:
                if remove_bkgd_method == 'BiRefNet':
                    _mask = birefnetrmbg.generate_mask(_image)
                else:
                    _mask = RMBG(_image)
                _mask = image2mask(_mask)
                _mask = 1 - _mask
            else:
                _mask = _masks[i] if i < len(_masks) else _masks[-1]

            if mask_grow != 0:
                _mask = expand_mask(_mask, mask_grow, 0)  # 扩张，模糊

            if color_of == 'subject':
                _mask = 1 - _mask

            if color_of == 'entire':
                blured_image = gaussian_blur(_image, int((_image.width + _image.height) / 400))
            else:
                _mask = tensor2pil(_mask)
                pixel_spread_image = pixel_spread(_image, _mask.convert('RGB'))
                blured_image = gaussian_blur(pixel_spread_image, int((_image.width + _image.height) / 400))
            if mode == 'main_color':
                ret_color = get_image_color_tone(blured_image)
            else:
                ret_color = get_image_color_average(blured_image)

            ret_image = Image.new('RGB', size=_image.size, color=ret_color)
            ret_images.append(pil2tensor(ret_image))
        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        hsv_color = RGB_to_HSV(Hex_to_RGB(ret_color))
        return (torch.cat(ret_images, dim=0), ret_color, hsv_color,)

```
