---
tags:
- Mask
---

# LayerMask: MaskGradient
## Documentation
- Class name: `LayerMask: MaskGradient`
- Category: `😺dzNodes/LayerMask`
- Output node: `False`

The `MaskGradient` node is designed to apply gradient effects to masks, enhancing visual aesthetics or highlighting specific areas. It leverages gradient parameters to dynamically adjust the appearance of input masks, offering a versatile tool for image manipulation and layer styling within graphical interfaces.
## Input types
### Required
- **`mask`**
    - The mask input represents the target image mask to which the gradient effect will be applied. It is crucial for defining the area of effect and the base for gradient application.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`invert_mask`**
    - This parameter determines whether the input mask should be inverted before applying the gradient effect, allowing for flexibility in targeting either the masked area or its inverse.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`gradient_side`**
    - Specifies the side of the mask where the gradient effect begins, enabling directional control over the gradient application.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`gradient_scale`**
    - Controls the scale of the gradient effect, affecting how the gradient transitions across the mask area.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`gradient_offset`**
    - Adjusts the starting point of the gradient effect within the mask, offering additional customization for the gradient's positioning.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`opacity`**
    - Sets the opacity level of the gradient effect, allowing for fine-tuning of the visual impact on the mask.
    - Comfy dtype: `INT`
    - Python dtype: `float`
### Optional
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output is a tensor containing the modified mask(s) with the applied gradient effect, showcasing the visual transformation.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskGradient:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        side = ['top', 'bottom', 'left', 'right']
        return {
            "required": {
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "gradient_side": (side,),
                "gradient_scale": ("INT", {"default": 100, "min": 1, "max": 9999, "step": 1}),
                "gradient_offset": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'mask_gradient'
    CATEGORY = '😺dzNodes/LayerMask'

    def mask_gradient(self, mask, invert_mask, gradient_side, gradient_scale, gradient_offset, opacity, ):


        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)

        l_masks = []
        ret_masks = []

        for m in mask:
            if invert_mask:
                m = 1 - m
            l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        for i in range(len(l_masks)):
            _mask = l_masks[i]
            _canvas = copy.copy(_mask)
            width = _mask.width
            height = _mask.height
            _gradient = gradient('#000000', '#FFFFFF',
                                 1024, 1024, 0)
            # (box_x, box_y, box_width, box_height) = min_bounding_rect(_mask)
            (box_x, box_y, box_width, box_height) = mask_area(_mask)
            log(f"{NODE_NAME}: Box detected. x={box_x},y={box_y},width={box_width},height={box_height}")
            if box_width < 1 or box_height < 1:
                log(f"Error: {NODE_NAME} skipped, because the mask is does'nt have valid area", message_type='error')
                return (mask,)

            if gradient_side == 'top':
                boxsize = (width, box_height)
                _gradient = _gradient.transpose(Image.FLIP_TOP_BOTTOM)
                _gradient = _gradient.resize(boxsize)
                _black = Image.new('RGB', size = boxsize, color = 'black')
                if gradient_scale != 100:
                    _box = Image.new('RGB', size = boxsize, color = 'black')
                    _gradient = _gradient.resize((width, int(box_height * gradient_scale / 100)))
                    _box.paste(_gradient, box = (0, 0))
                    _gradient = _box
                if gradient_offset != 0:
                    _box = Image.new('RGB', size = boxsize, color = 'black')
                    _boxwhite = Image.new('RGB', size = boxsize, color = 'white')
                    _box.paste(_gradient, box=(0, gradient_offset))
                    _box.paste(_boxwhite, box = (0, gradient_offset - _box.height))
                    _gradient = _box
                    if gradient_offset > box_height:
                        _gradient = Image.new('RGB', size = boxsize, color = 'white')
                _canvas.paste(_black, box = (0, box_y), mask = _gradient.convert('L'))
            elif gradient_side == 'bottom':
                boxsize = (width, box_height)
                _gradient = _gradient.resize((width, box_height))
                _black = Image.new('RGB', size = boxsize, color = 'black')
                if gradient_scale != 100:
                    _box = Image.new('RGB', size = boxsize, color = 'black')
                    _gradient = _gradient.resize((width, int(box_height * gradient_scale / 100)))
                    _box.paste(_gradient, box = (0, box_height - _gradient.height))
                    _gradient = _box
                if gradient_offset != 0:
                    _box = Image.new('RGB', size = boxsize, color = 'black')
                    _boxwhite = Image.new('RGB', size = boxsize, color = 'white')
                    _box.paste(_gradient, box=(0, gradient_offset))
                    _box.paste(_boxwhite, box = (0, gradient_offset + _box.height))
                    _gradient = _box
                    if gradient_offset < -box_height:
                        _gradient = Image.new('RGB', size=boxsize, color='white')
                _canvas.paste(_black, box = (0, box_y + 1), mask = _gradient.convert('L'))
            elif gradient_side == 'left':
                boxsize = (box_width, height)
                _gradient = _gradient.transpose(Image.ROTATE_270)
                _gradient = _gradient.resize(boxsize)
                _black = Image.new('RGB', size = boxsize, color = 'black')
                if gradient_scale != 100:
                    _box = Image.new('RGB', size = boxsize, color = 'black')
                    _gradient = _gradient.resize((int(box_width * gradient_scale / 100), height))
                    _box.paste(_gradient, box = (0, 0))
                    _gradient = _box
                if gradient_offset != 0:
                    _box = Image.new('RGB', size = boxsize, color = 'black')
                    _boxwhite = Image.new('RGB', size = boxsize, color = 'white')
                    _box.paste(_gradient, box=(gradient_offset, 0))
                    _box.paste(_boxwhite, box = (gradient_offset - _box.width, 0))
                    _gradient = _box
                    if gradient_offset > box_width:
                        _gradient = Image.new('RGB', size=boxsize, color='white')
                _canvas.paste(_black, box = (box_x, 0), mask = _gradient.convert('L'))
            elif gradient_side == 'right':
                boxsize = (box_width, height)
                _gradient = _gradient.transpose(Image.ROTATE_90)
                _gradient = _gradient.resize(boxsize)
                _black = Image.new('RGB', size = boxsize, color = 'black')
                if gradient_scale != 100:
                    _box = Image.new('RGB', size = boxsize, color = 'black')
                    _gradient = _gradient.resize((int(box_width * gradient_scale / 100), height))
                    _box.paste(_gradient, box = (box_width - _gradient.width, 0))
                    _gradient = _box
                if gradient_offset != 0:
                    _box = Image.new('RGB', size = boxsize, color = 'black')
                    _boxwhite = Image.new('RGB', size = boxsize, color = 'white')
                    _box.paste(_gradient, box=(gradient_offset, 0))
                    _box.paste(_boxwhite, box = (gradient_offset + _box.width, 0))
                    _gradient = _box
                    if gradient_offset < -box_width:
                        _gradient = Image.new('RGB', size=boxsize, color='white')

                _canvas.paste(_black, box = (box_x + 1, 0), mask = _gradient.convert('L'))
            # opacity
            if opacity < 100:
                _canvas = chop_image(_mask, _canvas, 'normal', opacity)
            ret_masks.append(image2mask(_canvas))

        log(f"{NODE_NAME} Processed {len(ret_masks)} mask(s).", message_type='finish')
        return (torch.cat(ret_masks, dim=0),)

```
