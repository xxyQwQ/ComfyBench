# LayerStyle: InnerGlow
## Documentation
- Class name: `LayerStyle: InnerGlow`
- Category: `😺dzNodes/LayerStyle`
- Output node: `False`

The InnerGlow node applies an inner glow effect to images, enhancing visuals by adding a soft light from within the subject. This effect can be used to create depth or highlight elements in a composition, contributing to the overall aesthetic appeal.
## Input types
### Required
- **`background_image`**
    - The 'background_image' input serves as the base layer onto which the inner glow effect is applied, providing the context and foundation for the effect.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`layer_image`**
    - The 'layer_image' input specifies the target layer within the background image to which the inner glow effect will be applied, allowing for precise control over the effect's placement.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`invert_mask`**
    - The 'invert_mask' input determines whether the mask applied to the layer image should be inverted, affecting how the inner glow effect interacts with the image's transparency.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`blend_mode`**
    - The 'blend_mode' input allows for the selection of blending modes that determine how the inner glow effect combines with the underlying images, offering various visual outcomes.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`opacity`**
    - The 'opacity' input controls the transparency level of the inner glow effect, enabling fine-tuning of its visual impact on the final image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`brightness`**
    - The 'brightness' input adjusts the intensity of the inner glow effect, influencing the overall luminosity and visibility of the effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`glow_range`**
    - The 'glow_range' input specifies the spread of the inner glow effect, determining how far the effect extends from the edges of the layer image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`blur`**
    - The 'blur' input defines the level of blur applied to the inner glow effect, contributing to the softness and diffusion of the light.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`light_color`**
    - The 'light_color' input sets the color of the light at the center of the inner glow effect, allowing for customization of the effect's hue.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`glow_color`**
    - The 'glow_color' input determines the color of the outer edge of the inner glow effect, enabling the creation of a gradient or solid color glow.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`layer_mask`**
    - The 'layer_mask' input (optional) provides an additional masking layer to further refine where the inner glow effect is applied, enhancing the precision of the effect's application.
    - Comfy dtype: `MASK`
    - Python dtype: `Mask`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output 'image' showcases the enhanced visual appeal of the background image with the applied inner glow effect, reflecting the successful integration of the effect.
    - Python dtype: `Image`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class InnerGlow:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        chop_mode = ['screen', 'add', 'lighter', 'normal', 'multply', 'subtract','difference','darker',
                     'color_burn', 'color_dodge', 'linear_burn', 'linear_dodge', 'overlay',
                     'soft_light', 'hard_light', 'vivid_light', 'pin_light', 'linear_light', 'hard_mix']
        return {
            "required": {
                "background_image": ("IMAGE", ),  #
                "layer_image": ("IMAGE",),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blend_mode": (chop_mode,),  # 混合模式
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
                "brightness": ("INT", {"default": 5, "min": 2, "max": 20, "step": 1}),  # 迭代
                "glow_range": ("INT", {"default": 48, "min": -9999, "max": 9999, "step": 1}),  # 扩张
                "blur": ("INT", {"default": 25, "min": 0, "max": 9999, "step": 1}),  # 扩张
                "light_color": ("STRING", {"default": "#FFBF30"}),  # 光源中心颜色
                "glow_color": ("STRING", {"default": "#FE0000"}),  # 辉光外围颜色
            },
            "optional": {
                "layer_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'inner_glow'
    CATEGORY = '😺dzNodes/LayerStyle'

    def inner_glow(self, background_image, layer_image,
                  invert_mask, blend_mode, opacity,
                  brightness, glow_range, blur, light_color, glow_color,
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

            blur_factor = blur / 20.0
            grow = glow_range
            inner_mask = _mask
            for x in range(brightness):
                blur = int(grow * blur_factor)
                _color = step_color(glow_color, light_color, brightness, x)
                glow_mask = expand_mask(image2mask(inner_mask), -grow, blur)  #扩张，模糊
                # 合成glow
                color_image = Image.new("RGB", _layer.size, color=_color)
                alpha = tensor2pil(mask_invert(glow_mask)).convert('L')
                _glow = chop_image(_layer, color_image, blend_mode, int(step_value(1, opacity, brightness, x)))
                _layer.paste(_glow, mask=alpha)
                grow = grow - int(glow_range/brightness)
            # 合成layer
            _layer.paste(_canvas, mask=ImageChops.invert(_mask))
            ret_images.append(pil2tensor(_layer))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
