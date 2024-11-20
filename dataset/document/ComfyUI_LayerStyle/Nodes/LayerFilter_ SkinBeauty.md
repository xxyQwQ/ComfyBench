---
tags:
- VisualEffects
---

# LayerFilter: SkinBeauty
## Documentation
- Class name: `LayerFilter: SkinBeauty`
- Category: `😺dzNodes/LayerFilter`
- Output node: `False`

The SkinBeauty node enhances the visual quality of skin in images by applying smoothing, adjusting brightness, and improving overall skin texture. It utilizes advanced image processing techniques to selectively blur, adjust thresholds, and manage opacity to achieve a more aesthetically pleasing and natural-looking skin appearance.
## Input types
### Required
- **`image`**
    - The input image or images to be processed for skin beauty enhancement. This is the primary data on which the skin beauty effects will be applied.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`smooth`**
    - Determines the degree of skin smoothing applied to the image. Higher values result in smoother skin.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`threshold`**
    - Adjusts the brightness threshold for highlighting effects. This parameter helps in enhancing the skin's high points by making them brighter or more subdued.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`opacity`**
    - Controls the opacity of the applied skin beauty effects, allowing for fine-tuning of the final image's look by blending the processed layer with the original.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The processed image with enhanced skin beauty effects applied.
    - Python dtype: `torch.Tensor`
- **`beauty_mask`**
    - Comfy dtype: `MASK`
    - A mask that highlights the areas of the image where skin beauty effects have been applied.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SkinBeauty:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),
                "smooth": ("INT", {"default": 20, "min": 1, "max": 64, "step": 1}),  # 磨皮程度
                "threshold": ("INT", {"default": -10, "min": -255, "max": 255, "step": 1}),  # 高光阈值
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "beauty_mask")
    FUNCTION = 'skin_beauty'
    CATEGORY = '😺dzNodes/LayerFilter'

    def skin_beauty(self, image, smooth, threshold, opacity
                  ):

        ret_images = []
        ret_masks = []
        for i in image:
            i = torch.unsqueeze(i, 0)
            _canvas = tensor2pil(i).convert('RGB')
            _R, _, _, _ = image_channel_split(_canvas, mode='RGB')
            _otsumask = gray_threshold(_R, otsu=True)
            _removebkgd = remove_background(_R, _otsumask, '#000000')
            auto_threshold = get_image_bright_average(_removebkgd) - 16
            light_mask = gray_threshold(_canvas, auto_threshold + threshold)
            blur = int((_canvas.width + _canvas.height) / 2000 * smooth)
            _image = image_beauty(_canvas, level=smooth)
            _image = gaussian_blur(_image, blur)
            _image = chop_image(_canvas, _image, 'normal', opacity)
            light_mask = gaussian_blur(light_mask, blur).convert('L')
            _canvas.paste(_image, mask=light_mask)

            ret_images.append(pil2tensor(_canvas))
            ret_masks.append(image2mask(light_mask))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

```
