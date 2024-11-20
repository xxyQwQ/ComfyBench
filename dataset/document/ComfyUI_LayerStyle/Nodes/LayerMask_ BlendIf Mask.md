---
tags:
- Mask
---

# LayerMask: BlendIf Mask
## Documentation
- Class name: `LayerMask: BlendIf Mask`
- Category: `😺dzNodes/LayerMask`
- Output node: `False`

The `BlendIfMask` node provides functionality for blending images based on specific color channel thresholds, allowing for complex masking operations that can dynamically adjust the visibility of layers or effects based on the underlying color values.
## Input types
### Required
- **`image`**
    - The primary image to be processed, serving as the base for the blend if operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`invert_mask`**
    - Determines whether the mask should be inverted, affecting how the blend conditions are applied.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`blend_if`**
    - Specifies the color channel (gray, red, green, blue) to be used for the blend if condition, dictating how the blending is controlled.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`black_point`**
    - The threshold below which pixels are fully transparent in the blending operation, allowing for fine-tuning of dark areas.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`black_range`**
    - Defines the range over which the transparency effect fades from fully transparent to fully opaque, offering control over the blend transition in dark areas.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`white_point`**
    - The threshold above which pixels are fully opaque, enabling precise control over bright areas in the blend.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`white_range`**
    - Determines the range over which the opacity effect transitions from fully opaque to fully transparent, facilitating smooth blending in bright areas.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`mask`**
    - An optional mask that can be applied to further refine the blending operation, providing additional control over which parts of the image are affected.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The resulting mask after applying the blend if conditions, which can be used to selectively apply effects or layer visibility.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class BlendIfMask:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        blend_if_list = ["gray", "red", "green", "blue"]
        return {
            "required": {
                "image": ("IMAGE", ),
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blend_if": (blend_if_list,),
                "black_point": ("INT", {"default": 0, "min": 0, "max": 254, "step": 1, "display": "slider"}),
                "black_range": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "white_point": ("INT", {"default": 255, "min": 1, "max": 255, "step": 1, "display": "slider"}),
                "white_range": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'blend_if_mask'
    CATEGORY = '😺dzNodes/LayerMask'

    def blend_if_mask(self, image, invert_mask, blend_if,
                      black_point, black_range,
                      white_point, white_range,
                      mask=None
                      ):


        ret_masks = []
        input_images = []
        input_masks = []

        for i in image:
            input_images.append(torch.unsqueeze(i, 0))
            m = tensor2pil(i)
            if m.mode == 'RGBA':
                input_masks.append(m.split()[-1])
            else:
                input_masks.append(Image.new('L', size=m.size, color='white'))
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            input_masks = []
            for m in mask:
                if invert_mask:
                    m = 1 - m
                input_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))
        max_batch = max(len(input_images), len(input_masks))

        for i in range(max_batch):
            _image = input_images[i] if i < len(input_images) else input_images[-1]
            _image = tensor2pil(_image).convert('RGB')

            if blend_if == "red":
                gray = image_channel_split(_image, 'RGB')[0]
            elif blend_if == "green":
                gray = image_channel_split(_image, 'RGB')[1]
            elif blend_if == "blue":
                gray = image_channel_split(_image, 'RGB')[2]
            else:
                gray = _image.convert('L')

            _mask = input_masks[i] if i < len(input_masks) else input_masks[-1]

            gray = histogram_range(gray, black_point, black_range, white_point, white_range)
            black = Image.new('L', size=_image.size, color='black')
            _mask = ImageChops.invert(_mask)
            gray.paste(black, mask=_mask)

            ret_masks.append(image2mask(gray))

        log(f"{NODE_NAME} Processed {len(ret_masks)} image(s).", message_type='finish')
        return (torch.cat(ret_masks, dim=0),)

```
