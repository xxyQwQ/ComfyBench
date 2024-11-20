# LayerStyle: DropShadow
## Documentation
- Class name: `LayerStyle: DropShadow`
- Category: `😺dzNodes/LayerStyle`
- Output node: `False`

The DropShadow node applies a drop shadow effect to an image layer, enhancing its visual depth by simulating a shadow cast behind it. This node processes input images to add shadows based on specified parameters such as distance, direction, color, and blur, effectively creating a more dynamic and visually appealing composition.
## Input types
### Required
- **`background_image`**
    - The background image over which the drop shadow effect will be applied. It serves as the canvas for the shadow and layer composition.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`layer_image`**
    - The image layer to which the drop shadow effect will be applied. This layer will appear to cast a shadow on the background image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`invert_mask`**
    - A boolean flag to indicate whether the mask should be inverted. This affects how the shadow is applied relative to the mask, allowing for more control over the shadow's appearance.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`blend_mode`**
    - Specifies the blending mode to be used when applying the shadow, affecting how the shadow merges with the layer.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`opacity`**
    - The opacity level of the drop shadow, determining how transparent the shadow appears over the background.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`distance_x`**
    - The horizontal distance to offset the shadow from the layer, simulating the direction of light.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`distance_y`**
    - The vertical distance to offset the shadow from the layer, further contributing to the illusion of depth.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`grow`**
    - The amount by which the shadow expands beyond the original boundaries of the layer, making the shadow appear larger or smaller.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`blur`**
    - The blur radius for the shadow, determining how sharp or soft the shadow edges are.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`shadow_color`**
    - The color of the drop shadow, allowing customization to match the aesthetic of the composition.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`layer_mask`**
    - An optional mask that defines the area of the layer image where the drop shadow effect should be applied. It helps in fine-tuning the shadow effect by restricting it to specific parts of the layer image.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting image after applying the drop shadow effect, combining the background, layer, and the generated shadow into a single image.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class DropShadow:

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
    FUNCTION = 'drop_shadow'
    CATEGORY = '😺dzNodes/LayerStyle'

    def drop_shadow(self, background_image, layer_image,
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
            _shadow = chop_image(_canvas, shadow_color, blend_mode, opacity)
            _canvas.paste(_shadow, mask=alpha)
            # 合成layer
            _canvas.paste(_layer, mask=_mask)

            ret_images.append(pil2tensor(_canvas))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
