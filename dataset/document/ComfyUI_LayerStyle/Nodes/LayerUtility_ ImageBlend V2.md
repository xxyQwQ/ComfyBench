---
tags:
- Image
- ImageBlend
- ImageComposite
---

# LayerUtility: ImageBlend V2
## Documentation
- Class name: `LayerUtility: ImageBlend V2`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

This node facilitates the blending of two images, offering a range of blending modes and control over the blend's opacity and mask inversion. It's designed to enhance creative workflows in image processing by allowing for complex visual effects through layer blending.
## Input types
### Required
- **`background_image`**
    - Serves as the base layer for blending, providing the foundational imagery over which the layer image is blended.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`layer_image`**
    - Acts as the top layer in the blend, overlaying the background image according to the specified blending mode and opacity.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`invert_mask`**
    - Determines whether the mask applied to the layer image should be inverted, affecting how the blend is masked out.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`blend_mode`**
    - Specifies the method of blending the layer image with the background, such as multiply or overlay, influencing the visual outcome.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`opacity`**
    - Controls the transparency level of the layer image over the background, allowing for fine-tuning of the blend intensity.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`layer_mask`**
    - Optional mask that can be applied to the layer image, defining areas of transparency for intricate blending effects.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The result of blending the layer image with the background image, reflecting the chosen blend mode, opacity, and mask settings.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ImageBlendV2:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "background_image": ("IMAGE", ),  #
                "layer_image": ("IMAGE",),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blend_mode": (chop_mode_v2,),  # 混合模式
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
            },
            "optional": {
                "layer_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'image_blend_v2'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_blend_v2(self, background_image, layer_image,
                  invert_mask, blend_mode, opacity,
                  layer_mask=None
                  ):

        b_images = []
        l_images = []
        l_masks = []
        ret_images = []
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

            _canvas = tensor2pil(background_image).convert('RGB')
            _layer = tensor2pil(layer_image).convert('RGB')

            if _mask.size != _layer.size:
                _mask = Image.new('L', _layer.size, 'white')
                log(f"Warning: {NODE_NAME} mask mismatch, dropped!", message_type='warning')

            # 合成layer
            _comp = chop_image_v2(_canvas, _layer, blend_mode, opacity)
            _canvas.paste(_comp, mask=_mask)

            ret_images.append(pil2tensor(_canvas))
        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
