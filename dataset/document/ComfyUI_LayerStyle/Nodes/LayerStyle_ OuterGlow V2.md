# LayerStyle: OuterGlow V2
## Documentation
- Class name: `LayerStyle: OuterGlow V2`
- Category: `😺dzNodes/LayerStyle`
- Output node: `False`

The OuterGlow V2 node is designed to enhance images by applying a sophisticated outer glow effect. This node allows for detailed customization of the glow through various parameters, enabling users to adjust aspects such as intensity, spread, and color to achieve the desired visual effect.
## Input types
### Required
- **`background_image`**
    - The base image over which the outer glow effect will be applied. It serves as the canvas for the glow effect.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`layer_image`**
    - The image layer to which the outer glow effect will be applied. This layer is the primary focus of the glow enhancement.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`invert_mask`**
    - A boolean parameter that determines whether the mask applied to the layer image should be inverted, affecting the area of the glow effect.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`blend_mode`**
    - Specifies the blending mode to be used for combining the glow effect with the layer image, influencing the overall appearance of the effect.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`opacity`**
    - Controls the opacity of the outer glow effect, allowing for fine-tuning of its visibility.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`brightness`**
    - Adjusts the brightness of the glow effect, affecting its intensity and visibility.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`glow_range`**
    - Defines the range of the glow effect, determining how far the glow extends from the layer image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`blur`**
    - Sets the level of blur applied to the glow effect, contributing to its softness and diffusion.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`light_color`**
    - The color of the light source at the center of the glow effect, contributing to the effect's overall color scheme.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`glow_color`**
    - The color of the outermost part of the glow effect, defining its peripheral appearance.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`layer_mask`**
    - An optional mask that can be applied to the layer image to control the application of the glow effect to specific areas.
    - Comfy dtype: `MASK`
    - Python dtype: `Mask`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting image after the outer glow effect has been applied, showcasing the enhanced visual appeal.
    - Python dtype: `Image`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class OuterGlowV2:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        modes = copy.copy(BLEND_MODES)
        chop_mode_list = ["screen", "linear dodge(add)", "color dodge", "lighten", "dodge", "hard light", "linear light"]
        for i in chop_mode_list:
            modes.pop(i)
        chop_mode_list.extend(list(modes.keys()))

        return {
            "required": {
                "background_image": ("IMAGE", ),  #
                "layer_image": ("IMAGE",),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "blend_mode": (chop_mode_list,),  # 混合模式
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
    FUNCTION = 'outer_glow_v2'
    CATEGORY = '😺dzNodes/LayerStyle'

    def outer_glow_v2(self, background_image, layer_image,
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
        blur_factor = blur / 20.0
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
            grow = glow_range
            for x in range(brightness):
                blur = int(grow * blur_factor)
                _color = step_color(glow_color, light_color, brightness, x)
                glow_mask = expand_mask(image2mask(_mask), grow, blur)  #扩张，模糊
                # 合成glow
                color_image = Image.new("RGB", _layer.size, color=_color)
                alpha = tensor2pil(glow_mask).convert('L')
                _glow = chop_image_v2(_canvas, color_image, blend_mode, int(step_value(1, opacity, brightness, x)))
                _canvas.paste(_glow.convert('RGB'), mask=alpha)
                grow = grow - int(glow_range/brightness)
            # 合成layer
            _canvas.paste(_layer, mask=_mask)

            ret_images.append(pil2tensor(_canvas))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
