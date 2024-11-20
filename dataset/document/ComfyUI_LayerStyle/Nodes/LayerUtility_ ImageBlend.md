# LayerUtility: ImageBlend
## Documentation
- Class name: `LayerUtility: ImageBlend`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

This node is designed to blend two images together using a variety of blending modes. It leverages predefined functions to apply different mathematical operations for combining the pixel values of the background and layer images, resulting in a composite image that can be used for various visual effects or image processing tasks.
## Input types
### Required
- **`background_image`**
    - The background image over which the layer image will be blended. This forms the base of the composite image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`layer_image`**
    - The layer image to be blended onto the background image. This image is combined with the background based on the specified blend mode and opacity.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`invert_mask`**
    - A boolean parameter that determines whether the mask applied to the layer image should be inverted, affecting how the blend is applied.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`blend_mode`**
    - Specifies the method of blending the layer image with the background. Different modes can produce various visual effects.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`opacity`**
    - Controls the transparency of the layer image, allowing for finer control over the blend intensity.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`layer_mask`**
    - An optional mask that can be applied to the layer image, defining which parts of the layer should be visible in the final blend.
    - Comfy dtype: `MASK`
    - Python dtype: `Mask`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The result of blending the layer image with the background image according to the specified parameters.
    - Python dtype: `Image`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ImageBlend:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "background_image": ("IMAGE", ),  #
                "layer_image": ("IMAGE",),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blend_mode": (chop_mode,),  # 混合模式
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
            },
            "optional": {
                "layer_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'image_blend'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_blend(self, background_image, layer_image,
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
            _comp = chop_image(_canvas, _layer, blend_mode, opacity)
            _canvas.paste(_comp, mask=_mask)

            ret_images.append(pil2tensor(_canvas))
        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
