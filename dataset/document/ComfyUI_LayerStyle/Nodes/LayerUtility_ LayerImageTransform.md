---
tags:
- Image
- ImageTransformation
---

# LayerUtility: LayerImageTransform
## Documentation
- Class name: `LayerUtility: LayerImageTransform`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

The LayerImageTransform node is designed to perform various image transformations on a given layer, including scaling, rotating, mirroring, and applying anti-aliasing. It allows for precise adjustments to an image's appearance, enabling the creation of customized visual effects and adjustments.
## Input types
### Required
- **`image`**
    - The input image to be transformed. It serves as the base for all subsequent transformations, including scaling, rotating, and mirroring.
    - Comfy dtype: `IMAGE`
    - Python dtype: `IMAGE`
- **`x`**
    - The horizontal offset to apply after transformations, allowing for precise positioning of the transformed image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`y`**
    - The vertical offset to apply after transformations, enabling precise vertical positioning of the transformed image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`mirror`**
    - Specifies the mirroring mode to apply to the image, such as horizontal or vertical flipping, enhancing the image's symmetry or creating reflections.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scale`**
    - The scaling factor to apply to the image, adjusting its size while preserving its aspect ratio.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`aspect_ratio`**
    - The aspect ratio to maintain during scaling, allowing for proportional adjustments to the image's dimensions.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`rotate`**
    - The rotation angle in degrees to apply to the image, enabling orientation adjustments.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`transform_method`**
    - The method used for image transformation, such as resizing or rotating, affecting the image's quality and appearance.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`anti_aliasing`**
    - The level of anti-aliasing to apply, reducing the appearance of jagged edges in the transformed image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The transformed image after applying scaling, rotating, mirroring, and anti-aliasing effects.
    - Python dtype: `IMAGE`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LayerImageTransform:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        mirror_mode = ['None', 'horizontal', 'vertical']
        method_mode = ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest']
        return {
            "required": {
                "image": ("IMAGE",),  #
                "x": ("INT", {"default": 0, "min": -99999, "max": 99999, "step": 1}),
                "y": ("INT", {"default": 0, "min": -99999, "max": 99999, "step": 1}),
                "mirror": (mirror_mode,),  # 镜像翻转
                "scale": ("FLOAT", {"default": 1, "min": 0.01, "max": 100, "step": 0.01}),
                "aspect_ratio": ("FLOAT", {"default": 1, "min": 0.01, "max": 100, "step": 0.01}),
                "rotate": ("FLOAT", {"default": 0, "min": -999999, "max": 999999, "step": 0.01}),
                "transform_method": (method_mode,),
                "anti_aliasing": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'layer_image_transform'
    CATEGORY = '😺dzNodes/LayerUtility'

    def layer_image_transform(self, image, x, y, mirror, scale, aspect_ratio, rotate,
                            transform_method, anti_aliasing,
                  ):

        l_images = []
        l_masks = []
        ret_images = []

        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])

        for i in range(len(l_images)):
            layer_image = l_images[i] if i < len(l_images) else l_images[-1]
            _image = tensor2pil(layer_image).convert('RGB')
            if i < len(l_masks):
                _mask = l_masks[i]
            else:
                _mask = Image.new('L', size=_image.size, color='white')
            _image_canvas = Image.new('RGB', size=_image.size, color='black')
            _mask_canvas = Image.new('L', size=_mask.size, color='black')
            orig_layer_width = _image.width
            orig_layer_height = _image.height
            target_layer_width = int(orig_layer_width * scale)
            target_layer_height = int(orig_layer_height * scale * aspect_ratio)
            # mirror
            if mirror == 'horizontal':
                _image = _image.transpose(Image.FLIP_LEFT_RIGHT)
                _mask = _mask.transpose(Image.FLIP_LEFT_RIGHT)
            elif mirror == 'vertical':
                _image = _image.transpose(Image.FLIP_TOP_BOTTOM)
                _mask = _mask.transpose(Image.FLIP_TOP_BOTTOM)
            # scale
            _image = _image.resize((target_layer_width, target_layer_height))
            _mask = _mask.resize((target_layer_width, target_layer_height))
            # rotate
            _image, _mask, _ = image_rotate_extend_with_alpha(_image, rotate, _mask, transform_method, anti_aliasing)
            # composit layer
            paste_x = (orig_layer_width - _image.width) // 2 + x
            paste_y = (orig_layer_height - _image.height) // 2 + y
            _image_canvas.paste(_image, (paste_x, paste_y))
            _mask_canvas.paste(_mask, (paste_x, paste_y))
            if tensor2pil(layer_image).mode == 'RGBA':
                _image_canvas = RGB2RGBA(_image_canvas, _mask_canvas)

            ret_images.append(pil2tensor(_image_canvas))

        log(f"{NODE_NAME} Processed {len(l_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
