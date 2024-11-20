# LayerStyle: DropShadow V2
## Documentation
- Class name: `LayerStyle: DropShadow V2`
- Category: `😺dzNodes/LayerStyle`
- Output node: `False`

The DropShadow V2 node applies an advanced drop shadow effect to images, enhancing visual depth and realism. It processes input images by adding a customizable shadow based on parameters such as distance, blur, and color, effectively simulating the appearance of a shadow cast by the layer onto its background.
## Input types
### Required
- **`background_image`**
    - The background image over which the shadow and layer will be composited. It serves as the canvas for the drop shadow effect, providing context and depth to the layered image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`layer_image`**
    - The image of the layer to which the drop shadow effect will be applied. This image is composited over the background image, with the shadow effect enhancing its visual separation and depth.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`invert_mask`**
    - A boolean flag that indicates whether the mask should be inverted. This affects how the shadow is applied relative to the mask, allowing for more flexible shadow effects.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`blend_mode`**
    - Specifies the blending mode to be used when applying the shadow, affecting how the shadow merges with the layer and background images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`opacity`**
    - The opacity level of the shadow, determining how transparent the shadow appears over the background. It allows for fine-tuning the intensity of the shadow effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`distance_x`**
    - The horizontal distance by which the shadow is offset from the layer, simulating the direction and distance of the light source relative to the layer.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`distance_y`**
    - The vertical distance by which the shadow is offset from the layer, further contributing to the realism of the shadow effect by simulating elevation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`grow`**
    - Determines the extent to which the shadow expands beyond the original boundaries of the layer, simulating the diffusing effect of light.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`blur`**
    - The level of blur applied to the shadow, simulating the softness of the shadow's edges. This parameter affects the shadow's realism and how it blends with the background.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`shadow_color`**
    - The color of the shadow, allowing for customization of the shadow effect to match the aesthetic of the image or scene.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`layer_mask`**
    - An optional mask that defines the area of the layer image where the drop shadow should be applied. It helps in fine-tuning the shadow effect by specifying which parts of the layer image should cast a shadow.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting image after applying the drop shadow effect, combining the background, layer images, and the generated shadow into a single composited image.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class DropShadowV2:

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
                "distance_x": ("INT", {"default": 25, "min": -9999, "max": 9999, "step": 1}),  # x_偏移
                "distance_y": ("INT", {"default": 25, "min": -9999, "max": 9999, "step": 1}),  # y_偏移
                "grow": ("INT", {"default": 6, "min": -9999, "max": 9999, "step": 1}),  # 扩张
                "blur": ("INT", {"default": 18, "min": 0, "max": 100, "step": 1}),  # 模糊
                "shadow_color": ("STRING", {"default": "#000000"}),  # 背景颜色
            },
            "optional": {
                "layer_mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'drop_shadow_v2'
    CATEGORY = '😺dzNodes/LayerStyle'

    def drop_shadow_v2(self, background_image, layer_image,
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
            _layer = tensor2pil(layer_image)

            if _mask.size != _layer.size:
                _mask = Image.new('L', _layer.size, 'white')
                log(f"Warning: {NODE_NAME} mask mismatch, dropped!", message_type='warning')

            if distance_x != 0 or distance_y != 0:
                __mask = shift_image(_mask, distance_x, distance_y)  # 位移
            shadow_mask = expand_mask(image2mask(__mask), grow, blur)  #扩张，模糊
            # 合成阴影
            alpha = tensor2pil(shadow_mask).convert('L')
            _shadow = chop_image_v2(_canvas, shadow_color, blend_mode, opacity)
            _canvas.paste(_shadow, mask=alpha)
            # 合成layer
            _canvas.paste(_layer, mask=_mask)

            ret_images.append(pil2tensor(_canvas))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
