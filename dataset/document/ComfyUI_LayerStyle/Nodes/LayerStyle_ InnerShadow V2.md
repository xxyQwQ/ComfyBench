# LayerStyle: InnerShadow V2
## Documentation
- Class name: `LayerStyle: InnerShadow V2`
- Category: `😺dzNodes/LayerStyle`
- Output node: `False`

The InnerShadow V2 node is designed to apply an inner shadow effect to a given layer image, enhancing visual depth and dimensionality. It processes images by applying a shadow inside the layer boundaries based on specified parameters such as shadow color, distance, blur, and opacity, ensuring a more dynamic and visually appealing result.
## Input types
### Required
- **`background_image`**
    - The background image over which the layer image will be placed. It serves as the canvas for the inner shadow effect, ensuring that the shadow blends seamlessly with the background.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`layer_image`**
    - The layer image to which the inner shadow effect will be applied. This image is modified to include the shadow, creating a sense of depth within the layer itself.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`invert_mask`**
    - A boolean flag indicating whether the mask should be inverted. This affects how the shadow effect is applied relative to the masked areas.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`blend_mode`**
    - The blending mode used to combine the shadow with the layer, affecting how the shadow merges with the layer's colors and textures.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`opacity`**
    - The opacity of the inner shadow, controlling the transparency of the shadow effect for subtler or more pronounced visual impact.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`distance_x`**
    - The horizontal displacement of the shadow within the layer, allowing for the customization of the shadow's position relative to the layer.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`distance_y`**
    - The vertical displacement of the shadow within the layer, enabling the adjustment of the shadow's vertical position for a more tailored effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`grow`**
    - The amount by which the shadow's size is increased beyond the original layer boundaries, contributing to a more pronounced shadow effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`blur`**
    - The level of blur applied to the shadow, determining the softness of the shadow edges for a more natural and integrated appearance.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`shadow_color`**
    - The color of the inner shadow, allowing for the customization of the shadow to match or contrast with the layer and background for desired visual effects.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`layer_mask`**
    - An optional mask that defines the area of the layer image where the inner shadow effect should be applied, providing additional control over the effect's application.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting image after applying the inner shadow effect, showcasing the enhanced visual depth and dimensionality of the layer.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class InnerShadowV2:

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
                "opacity": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),  # 透明度
                "distance_x": ("INT", {"default": 5, "min": -9999, "max": 9999, "step": 1}),  # x_偏移
                "distance_y": ("INT", {"default": 5, "min": -9999, "max": 9999, "step": 1}),  # y_偏移
                "grow": ("INT", {"default": 2, "min": -9999, "max": 9999, "step": 1}),  # 扩张
                "blur": ("INT", {"default": 15, "min": 0, "max": 100, "step": 1}),  # 模糊
                "shadow_color": ("STRING", {"default": "#000000"}),  # 背景颜色
            },
            "optional": {
                "layer_mask": ("MASK",),  #
            }
        }


    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'inner_shadow_v2'
    CATEGORY = '😺dzNodes/LayerStyle'

    def inner_shadow_v2(self, background_image, layer_image,
                  invert_mask, blend_mode, opacity, distance_x, distance_y,
                  grow, blur, shadow_color,
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
        if layer_mask is not None:
            if layer_mask.dim() == 2:
                layer_mask = torch.unsqueeze(layer_mask, 0)
            l_masks = []
            for m in layer_mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))
        if len(l_masks) == 0:
            log(f"Error: {NODE_NAME} skipped, because the available mask is not found.", message_type='error')
            return (background_image,)
        max_batch = max(len(b_images), len(l_images), len(l_masks))
        distance_x = -distance_x
        distance_y = -distance_y
        shadow_color = Image.new("RGB", tensor2pil(l_images[0]).size, color=shadow_color)
        for i in range(max_batch):
            background_image = b_images[i] if i < len(b_images) else b_images[-1]
            layer_image = l_images[i] if i < len(l_images) else l_images[-1]
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]
            # preprocess
            _canvas = tensor2pil(background_image).convert('RGB')
            _layer = tensor2pil(layer_image).convert('RGB')
            if _mask.size != _layer.size:
                _mask = Image.new('L', _layer.size, 'white')
                log(f"Warning: {NODE_NAME} mask mismatch, dropped!", message_type='warning')

            if distance_x != 0 or distance_y != 0:
                __mask = shift_image(_mask, distance_x, distance_y)  # 位移
            shadow_mask = expand_mask(image2mask(__mask), grow, blur)  #扩张，模糊
            # 合成阴影
            alpha = tensor2pil(shadow_mask).convert('L')
            _shadow = chop_image_v2(_layer, shadow_color, blend_mode, opacity)
            _layer.paste(_shadow, mask=ImageChops.invert(alpha))
            # 合成layer
            _canvas.paste(_layer, mask=_mask)

            ret_images.append(pil2tensor(_canvas))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
