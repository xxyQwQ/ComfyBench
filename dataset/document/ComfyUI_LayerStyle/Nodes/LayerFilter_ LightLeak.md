---
tags:
- VisualEffects
---

# LayerFilter: LightLeak
## Documentation
- Class name: `LayerFilter: LightLeak`
- Category: `😺dzNodes/LayerFilter`
- Output node: `False`

The LightLeak node applies a light leak effect to images, simulating the accidental exposure of film to light, enhancing the visual appeal with a vintage or artistic flair. It allows customization of the light leak's appearance, including its position, hue, saturation, and opacity, to achieve a variety of visual effects.
## Input types
### Required
- **`image`**
    - The input image to which the light leak effect will be applied. This is the primary canvas for the effect.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`light`**
    - Specifies the type of light leak effect to apply, which can be a specific light pattern or a random selection from a predefined set.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`corner`**
    - Determines the corner of the image where the light leak effect will be primarily applied, allowing for directional control of the effect.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`hue`**
    - Adjusts the hue of the light leak effect, enabling color customization to match or contrast with the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`saturation`**
    - Modifies the saturation of the light leak effect, allowing for more vivid or subdued color intensity.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`opacity`**
    - Controls the opacity of the light leak effect, affecting its prominence and blend with the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image with the applied light leak effect, showcasing the enhanced visual appeal.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class LightLeak:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        light_list = ['random', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                      '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                      '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                      '31', '32']
        corner_list = ['left_top', 'right_top', 'left_bottom', 'right_bottom']
        return {
            "required": {
                "image": ("IMAGE", ),
                "light": (light_list,),
                "corner": (corner_list,),
                "hue": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "saturation": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1})
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'light_leak'
    CATEGORY = '😺dzNodes/LayerFilter'

    def light_leak(self, image, light, corner, hue, saturation, opacity):

        ret_images = []
        light_leak_images = load_light_leak_images()
        if light == 'random':
            random.seed(time.time())
            light_index = random.randint(0,31)
        else:
            light_index = int(light) - 1

        for i in image:
            i = torch.unsqueeze(i, 0)
            _canvas = tensor2pil(i).convert('RGB')
            _light = light_leak_images[light_index]
            if _canvas.width < _canvas.height:
                _light = _light.transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)
            if corner == 'right_top':
                _light = _light.transpose(Image.FLIP_LEFT_RIGHT)
            elif corner == 'left_bottom':
                _light = _light.transpose(Image.FLIP_TOP_BOTTOM)
            elif corner == 'right_bottom':
                _light = _light.transpose(Image.ROTATE_180)
            if hue != 0 or saturation != 0:
                _h, _s, _v = _light.convert('HSV').split()
                if hue != 0:
                    _h = image_hue_offset(_h, hue)
                if saturation != 0:
                    _s = image_gray_offset(_s, saturation)
                _light = image_channel_merge((_h, _s, _v), 'HSV')
            resize_sampler = Image.BILINEAR
            _light = fit_resize_image(_light, _canvas.width, _canvas.height, fit='crop', resize_sampler=resize_sampler)
            ret_image = chop_image(_canvas, _light, blend_mode=blend_mode, opacity = opacity)
            ret_images.append(pil2tensor(ret_image))
            
        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
