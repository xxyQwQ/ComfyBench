---
tags:
- Image
- ImageBlend
- ImageComposite
---

# LayerUtility: ImageBlendAdvance V2
## Documentation
- Class name: `LayerUtility: ImageBlendAdvance V2`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

This node specializes in advanced blending techniques for layer-based image editing. It leverages sophisticated algorithms to merge images, allowing for more complex and nuanced image compositions than its predecessor.
## Input types
### Required
- **`background_image`**
    - The primary image over which the layer image will be blended, serving as the backdrop for the composition.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`layer_image`**
    - The secondary image to be blended onto the background image, acting as the overlay in the composition.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`invert_mask`**
    - A boolean value that determines whether the mask applied to the layer image should be inverted, affecting the blending areas.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`blend_mode`**
    - Specifies the method of blending used to combine the background and layer images, influencing the visual outcome of the blend.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`opacity`**
    - An integer defining the opacity level of the layer image, dictating its transparency when blended with the background image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`x_percent`**
    - A float value indicating the horizontal positioning of the layer image relative to the background image, as a percentage.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`y_percent`**
    - A float value indicating the vertical positioning of the layer image relative to the background image, as a percentage.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`mirror`**
    - Specifies the mirror mode applied to the layer image, such as horizontal or vertical flipping, before blending.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scale`**
    - A float value that determines the scaling factor of the layer image, affecting its size relative to the background image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`aspect_ratio`**
    - A float value that adjusts the aspect ratio of the layer image, influencing its proportions when scaled.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`rotate`**
    - A float value specifying the rotation angle of the layer image, in degrees, applied before blending.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`transform_method`**
    - Defines the interpolation method used for scaling and rotating the layer image, affecting the quality of the transformation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`anti_aliasing`**
    - An integer indicating the level of anti-aliasing applied to the layer image, enhancing the smoothness of its edges.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`layer_mask`**
    - An optional mask that can be applied to the layer image, allowing for more precise control over which parts of the layer are visible or blended.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting image after blending the layer image with the background, incorporating all specified adjustments and effects.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The mask applied to the layer image, determining the areas of the layer that are visible or blended in the final composition.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageBlendAdvanceV2:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        mirror_mode = ['None', 'horizontal', 'vertical']
        method_mode = ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest']
        return {
            "required": {
                "background_image": ("IMAGE", ),  #
                "layer_image": ("IMAGE",),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blend_mode": (chop_mode_v2,),  # 混合模式
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
                "x_percent": ("FLOAT", {"default": 50, "min": -999, "max": 999, "step": 0.01}),
                "y_percent": ("FLOAT", {"default": 50, "min": -999, "max": 999, "step": 0.01}),
                "mirror": (mirror_mode,),  # 镜像翻转
                "scale": ("FLOAT", {"default": 1, "min": 0.01, "max": 100, "step": 0.01}),
                "aspect_ratio": ("FLOAT", {"default": 1, "min": 0.01, "max": 100, "step": 0.01}),
                "rotate": ("FLOAT", {"default": 0, "min": -999999, "max": 999999, "step": 0.01}),
                "transform_method": (method_mode,),
                "anti_aliasing": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
            },
            "optional": {
                "layer_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = 'image_blend_advance_v2'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_blend_advance_v2(self, background_image, layer_image,
                            invert_mask, blend_mode, opacity,
                            x_percent, y_percent,
                            mirror, scale, aspect_ratio, rotate,
                            transform_method, anti_aliasing,
                            layer_mask=None
                            ):
        b_images = []
        l_images = []
        l_masks = []
        ret_images = []
        ret_masks = []
        for b in background_image:
            b_images.append(torch.unsqueeze(b, 0))
        for l in layer_image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', m.size, 'white'))
        if layer_mask is not None:
            if layer_mask.dim() == 2:
                layer_mask = torch.unsqueeze(layer_mask, 0)
            l_masks = []
            for m in layer_mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        max_batch = max(len(b_images), len(l_images), len(l_masks))
        for i in range(max_batch):
            background_image = b_images[i] if i < len(b_images) else b_images[-1]
            layer_image = l_images[i] if i < len(l_images) else l_images[-1]
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]
            # preprocess
            _canvas = tensor2pil(background_image).convert('RGB')
            _layer = tensor2pil(layer_image)

            if _mask.size != _layer.size:
                _mask = Image.new('L', _layer.size, 'white')
                log(f"Warning: {NODE_NAME} mask mismatch, dropped!", message_type='warning')

            orig_layer_width = _layer.width
            orig_layer_height = _layer.height
            _mask = _mask.convert("RGB")

            target_layer_width = int(orig_layer_width * scale)
            target_layer_height = int(orig_layer_height * scale * aspect_ratio)

            # mirror
            if mirror == 'horizontal':
                _layer = _layer.transpose(Image.FLIP_LEFT_RIGHT)
                _mask = _mask.transpose(Image.FLIP_LEFT_RIGHT)
            elif mirror == 'vertical':
                _layer = _layer.transpose(Image.FLIP_TOP_BOTTOM)
                _mask = _mask.transpose(Image.FLIP_TOP_BOTTOM)

            # scale
            _layer = _layer.resize((target_layer_width, target_layer_height))
            _mask = _mask.resize((target_layer_width, target_layer_height))
            # rotate
            _layer, _mask, _ = image_rotate_extend_with_alpha(_layer, rotate, _mask, transform_method, anti_aliasing)

            # 处理位置
            x = int(_canvas.width * x_percent / 100 - _layer.width / 2)
            y = int(_canvas.height * y_percent / 100 - _layer.height / 2)

            # composit layer
            _comp = copy.copy(_canvas)
            _compmask = Image.new("RGB", _comp.size, color='black')
            _comp.paste(_layer, (x, y))
            _compmask.paste(_mask, (x, y))
            _compmask = _compmask.convert('L')
            _comp = chop_image_v2(_canvas, _comp, blend_mode, opacity)

            # composition background
            _canvas.paste(_comp, mask=_compmask)

            ret_images.append(pil2tensor(_canvas))
            ret_masks.append(image2mask(_compmask))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

```
