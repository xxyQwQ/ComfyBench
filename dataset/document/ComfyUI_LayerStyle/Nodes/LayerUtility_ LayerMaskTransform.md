---
tags:
- Mask
---

# LayerUtility: LayerMaskTransform
## Documentation
- Class name: `LayerUtility: LayerMaskTransform`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

The LayerMaskTransform node is designed to apply various transformations to image masks, including scaling, mirroring, rotating, and adjusting aspect ratios. It enables the customization of image masks for further processing or visualization, supporting a range of transformation methods to fit different requirements.
## Input types
### Required
- **`mask`**
    - The input mask to be transformed. It serves as the primary subject for all subsequent transformations, determining the base image mask that will undergo scaling, mirroring, rotating, and aspect ratio adjustments.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`x`**
    - The horizontal offset to apply when positioning the transformed mask onto the original canvas. It affects the final placement of the mask, allowing for precise control over its location.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`y`**
    - The vertical offset to apply when positioning the transformed mask onto the original canvas. Similar to the 'x' parameter, it enables fine-tuning of the mask's placement on the canvas.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`mirror`**
    - Specifies the mirroring mode to apply to the mask, which can be horizontal, vertical, or none. This transformation flips the mask along the specified axis, altering its orientation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scale`**
    - The scaling factor to apply to the mask, allowing for size adjustments. This parameter directly influences the overall dimensions of the transformed mask.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`aspect_ratio`**
    - The aspect ratio to maintain during the scaling process. Adjusting this parameter alters the shape of the mask, enabling custom aspect ratio transformations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`rotate`**
    - The degree of rotation to apply to the mask. This parameter controls the angular adjustment of the mask, enabling it to be rotated to a specific orientation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`transform_method`**
    - The method used for transforming the mask, including scaling and rotating. It determines the algorithm applied for the transformation, affecting the quality and appearance of the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`anti_aliasing`**
    - The level of anti-aliasing to apply during the transformation process. Higher values result in smoother edges but may increase processing time.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The transformed mask, resulting from the applied transformations such as scaling, mirroring, rotating, and aspect ratio adjustments. It represents the final output of the node, ready for further use or visualization.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class LayerMaskTransform:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        mirror_mode = ['None', 'horizontal', 'vertical']
        method_mode = ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest']
        return {
            "required": {
                "mask": ("MASK",),  #
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

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'layer_mask_transform'
    CATEGORY = '😺dzNodes/LayerUtility'

    def layer_mask_transform(self, mask, x, y, mirror, scale, aspect_ratio, rotate,
                            transform_method, anti_aliasing,
                  ):

        l_masks = []
        ret_masks = []

        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        for m in mask:
            l_masks.append(torch.unsqueeze(m, 0))
        for i in range(len(l_masks)):
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]
            _mask = tensor2pil(_mask).convert('L')
            _mask_canvas = Image.new('L', size=_mask.size, color='black')
            orig_width = _mask.width
            orig_height = _mask.height
            target_layer_width = int(orig_width * scale)
            target_layer_height = int(orig_height * scale * aspect_ratio)
            # mirror
            if mirror == 'horizontal':
                _mask = _mask.transpose(Image.FLIP_LEFT_RIGHT)
            elif mirror == 'vertical':
                _mask = _mask.transpose(Image.FLIP_TOP_BOTTOM)
            # scale
            _mask = _mask.resize((target_layer_width, target_layer_height))
            # rotate
            _, _mask, _ = image_rotate_extend_with_alpha(_mask.convert('RGB'), rotate, _mask, transform_method, anti_aliasing)
            paste_x = (orig_width - _mask.width) // 2 + x
            paste_y = (orig_height - _mask.height) // 2 + y
            # composit layer
            _mask_canvas.paste(_mask, (paste_x, paste_y))

            ret_masks.append(image2mask(_mask_canvas))

        log(f"{NODE_NAME} Processed {len(l_masks)} mask(s).", message_type='finish')
        return (torch.cat(ret_masks, dim=0),)

```
