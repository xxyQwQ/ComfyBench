---
tags:
- VisualEffects
---

# LayerFilter: SoftLight
## Documentation
- Class name: `LayerFilter: SoftLight`
- Category: `😺dzNodes/LayerFilter`
- Output node: `False`

The SoftLight node applies a soft lighting effect to images, enhancing their visual appeal by blending them with various lighting conditions. It utilizes advanced image processing techniques to adjust the brightness, contrast, and detail of images to simulate the effect of soft light, making it ideal for creating mood or atmosphere in visual content.
## Input types
### Required
- **`image`**
    - The primary image to which the soft light effect will be applied. This image serves as the base for the lighting adjustments.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`soft`**
    - Controls the intensity of the soft light effect, allowing for fine-tuning of the blur amount applied to the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`threshold`**
    - Determines the threshold for highlighting, affecting how light or dark areas are enhanced or subdued in the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`opacity`**
    - Adjusts the opacity of the soft light effect, enabling control over how strongly the effect is applied to the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The image after the soft light effect has been applied, showcasing enhanced lighting and visual appeal.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SoftLight:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "soft": ("FLOAT", {"default": 1, "min": 0.2, "max": 10, "step": 0.01}),  # 模糊
                "threshold": ("INT", {"default": -10, "min": -255, "max": 255, "step": 1}),  # 高光阈值
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'soft_light'
    CATEGORY = '😺dzNodes/LayerFilter'

    def soft_light(self, image, soft, threshold, opacity,):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            blend_mode = 'screen'
            _canvas = tensor2pil(i).convert('RGB')
            blur = int((_canvas.width + _canvas.height) / 200 * soft)
            _otsumask = gray_threshold(_canvas, otsu=True)
            _removebkgd = remove_background(_canvas, _otsumask, '#000000').convert('L')
            auto_threshold = get_image_bright_average(_removebkgd)
            light_mask = gray_threshold(_canvas, auto_threshold + threshold)
            highlight_mask = gray_threshold(_canvas, auto_threshold + (255 - auto_threshold) // 2 + threshold // 2)
            blurimage = gaussian_blur(_canvas, soft).convert('RGB')
            light = chop_image(_canvas, blurimage, blend_mode=blend_mode, opacity=opacity)
            highlight = chop_image(light, blurimage, blend_mode=blend_mode, opacity=opacity)
            _canvas.paste(highlight, mask=gaussian_blur(light_mask, blur * 2).convert('L'))
            _canvas.paste(highlight, mask=gaussian_blur(highlight_mask, blur).convert('L'))

            ret_images.append(pil2tensor(_canvas))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
