---
tags:
- Color
---

# LayerColor: Brightness & Contrast
## Documentation
- Class name: `LayerColor: Brightness & Contrast`
- Category: `😺dzNodes/LayerColor`
- Output node: `False`

This node is designed to adjust the brightness, contrast, and saturation of images. It processes each image individually, allowing for fine-tuned enhancements to image aesthetics by modifying these three key visual parameters.
## Input types
### Required
- **`image`**
    - The input image or images to be processed. This parameter is crucial for the node's operation as it directly influences the visual output by applying the specified adjustments.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`brightness`**
    - Determines the level of brightness adjustment to be applied to the input images. A pivotal parameter for altering the image's luminance.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`contrast`**
    - Specifies the degree of contrast adjustment. This parameter is essential for enhancing or reducing the difference between the darkest and lightest tones in the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`saturation`**
    - Controls the saturation level, affecting the intensity of the image's colors. It plays a significant role in the visual impact of the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output images after applying brightness, contrast, and saturation adjustments. This reflects the visual modifications made to the input images, showcasing the node's capability to enhance image aesthetics.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ColorCorrectBrightnessAndContrast:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "brightness": ("FLOAT", {"default": 1, "min": 0.0, "max": 3, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1, "min": 0.0, "max": 3, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1, "min": 0.0, "max": 3, "step": 0.01}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_correct_brightness_and_contrast'
    CATEGORY = '😺dzNodes/LayerColor'

    def color_correct_brightness_and_contrast(self, image, brightness, contrast, saturation):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i,0)
            __image = tensor2pil(i)
            ret_image = __image.convert('RGB')
            if brightness != 1:
                brightness_image = ImageEnhance.Brightness(ret_image)
                ret_image = brightness_image.enhance(factor=brightness)
            if contrast != 1:
                contrast_image = ImageEnhance.Contrast(ret_image)
                ret_image = contrast_image.enhance(factor=contrast)
            if saturation != 1:
                color_image = ImageEnhance.Color(ret_image)
                ret_image = color_image.enhance(factor=saturation)

            if __image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, __image.split()[-1])
            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
