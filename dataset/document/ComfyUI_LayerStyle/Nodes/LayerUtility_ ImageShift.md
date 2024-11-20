---
tags:
- VisualEffects
---

# LayerUtility: ImageShift
## Documentation
- Class name: `LayerUtility: ImageShift`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

The ImageShift node is designed to manipulate images by shifting them horizontally and/or vertically. It supports cyclic shifting, allowing images to wrap around edges, and can apply a border mask to the shifted images for enhanced visual effects. This node is useful for creating dynamic visual transformations and adjusting image compositions.
## Input types
### Required
- **`image`**
    - The primary image to be shifted. This input is essential for determining the base visual content that will undergo the shifting process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`shift_x`**
    - Specifies the horizontal shift distance for the image. Positive values shift the image to the right, while negative values shift it to the left.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`shift_y`**
    - Specifies the vertical shift distance for the image. Positive values shift the image downwards, while negative values shift it upwards.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cyclic`**
    - Determines whether the image shift should be cyclic, allowing the shifted parts of the image to wrap around to the opposite edge.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`background_color`**
    - Defines the background color to use for areas of the image that become unfilled due to the shift.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`border_mask_width`**
    - Sets the width of the border mask applied to the edges of the shifted image, enhancing visual separation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`border_mask_blur`**
    - Determines the blur intensity of the border mask, providing a smoother transition at the image edges.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`mask`**
    - An optional mask that can be applied to the image, specifying areas to be excluded or differently treated during the shift.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The shifted image, transformed according to the specified parameters.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - A mask generated during the shifting process, potentially useful for further image processing steps.
    - Python dtype: `torch.Tensor`
- **`border_mask`**
    - Comfy dtype: `MASK`
    - A border mask applied to the shifted image, enhancing its visual appearance with a defined edge.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageShift:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "shift_x": ("INT", {"default": 256, "min": -9999, "max": 9999, "step": 1}),
                "shift_y": ("INT", {"default": 256, "min": -9999, "max": 9999, "step": 1}),
                "cyclic": ("BOOLEAN", {"default": True}),  # 是否循环重复
                "background_color": ("STRING", {"default": "#000000"}),
                "border_mask_width": ("INT", {"default": 20, "min": 0, "max": 999, "step": 1}),
                "border_mask_blur": ("INT", {"default": 12, "min": 0, "max": 999, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK",)
    RETURN_NAMES = ("image", "mask", "border_mask")
    FUNCTION = 'image_shift'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_shift(self, image, shift_x, shift_y,
                    cyclic, background_color,
                    border_mask_width, border_mask_blur,
                    mask=None
                    ):

        ret_images = []
        ret_masks = []
        ret_border_masks = []

        l_images = []
        l_masks = []


        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', size=m.size, color='white'))
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            l_masks = []
            for m in mask:
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        shift_x, shift_y = -shift_x, -shift_y
        for i in range(len(l_images)):
            _image = l_images[i]
            _canvas = tensor2pil(_image).convert('RGB')
            _mask = l_masks[i] if len(l_masks) < i else l_masks[-1]
            _border =  Image.new('L', size=_canvas.size, color='black')
            _border = draw_border(_border, border_width=border_mask_width, color='#FFFFFF')
            _border = _border.resize(_canvas.size)
            _canvas = shift_image(_canvas, shift_x, shift_y, background_color=background_color, cyclic=cyclic)
            _mask = shift_image(_mask, shift_x, shift_y, background_color='#000000', cyclic=cyclic)
            _border = shift_image(_border, shift_x, shift_y, background_color='#000000', cyclic=cyclic)
            _border = gaussian_blur(_border, border_mask_blur)

            ret_images.append(pil2tensor(_canvas))
            ret_masks.append(image2mask(_mask))
            ret_border_masks.append(image2mask(_border))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0), torch.cat(ret_border_masks, dim=0),)

```
