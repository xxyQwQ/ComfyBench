---
tags:
- VisualEffects
---

# LayerFilter: Film
## Documentation
- Class name: `LayerFilter: Film`
- Category: `😺dzNodes/LayerFilter`
- Output node: `False`

The Film node applies a cinematic film effect to images, enhancing them with a specific aesthetic that emulates the look and feel of traditional film photography. This node is designed to transform digital images by applying various film-like attributes, such as color grading, grain, and contrast adjustments, to achieve a visually appealing, filmic quality.
## Input types
### Required
- **`image`**
    - The primary input for the Film node, representing the image to which the film effect will be applied. This parameter is crucial as it determines the base upon which all film-like transformations are performed.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`center_x`**
    - Specifies the horizontal center of the effect focus, influencing how certain effects like blur and vignette are centered on the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`center_y`**
    - Determines the vertical center of the effect focus, affecting the positioning of effects such as blur and vignette on the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`saturation`**
    - Controls the intensity of the image's colors, allowing for either a more vivid or subdued color palette.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`vignette_intensity`**
    - Adjusts the strength of the vignette effect, adding a darkened border that can focus attention towards the center of the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`grain_power`**
    - Sets the intensity of the film grain effect, adding a textured, analog feel to the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`grain_scale`**
    - Determines the scale of the grain effect, affecting how large or fine the grain particles appear.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`grain_sat`**
    - Adjusts the saturation of the grain effect, influencing how pronounced the grain appears in color.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`grain_shadows`**
    - Controls the visibility of grain in the darker areas of the image, allowing for more or less texture in shadows.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`grain_highs`**
    - Adjusts the grain effect in the highlights, modifying how grain affects brighter areas of the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`blur_strength`**
    - Sets the intensity of the blur effect, which can simulate depth of field or focus effects.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`blur_focus_spread`**
    - Controls the spread of the blur effect around the focus point, affecting the transition between focused and blurred areas.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`focal_depth`**
    - Determines the depth of the focal area, influencing how much of the image appears in focus.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`depth_map`**
    - An optional parameter that provides a depth map for more advanced depth-based effects like variable blur.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output of the Film node is an image that has been processed to emulate the appearance of film, incorporating adjustments to color, grain, and contrast to enhance its cinematic quality.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class Film:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "center_x": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1, "min": 0.01, "max": 3, "step": 0.01}),
                "vignette_intensity": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "grain_power": ("FLOAT", {"default": 0.15, "min": 0, "max": 1, "step": 0.01}),
                "grain_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10, "step": 0.1}),
                "grain_sat": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "grain_shadows": ("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step": 0.01}),
                "grain_highs": ("FLOAT", {"default": 0.2, "min": 0, "max": 1, "step": 0.01}),
                "blur_strength": ("INT", {"default": 90, "min": 0, "max": 256, "step": 1}),
                "blur_focus_spread": ("FLOAT", {"default": 2.2, "min": 0.1, "max": 8, "step": 0.1}),
                "focal_depth": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1, "step": 0.01}),
            },
            "optional": {
                "depth_map": ("IMAGE",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'film'
    CATEGORY = '😺dzNodes/LayerFilter'

    def film(self, image, center_x, center_y, saturation, vignette_intensity,
                  grain_power, grain_scale, grain_sat, grain_shadows, grain_highs,
                  blur_strength, blur_focus_spread, focal_depth,
                  depth_map=None
                  ):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            _canvas = tensor2pil(i).convert('RGB')

            if saturation != 1:
                color_image = ImageEnhance.Color(_canvas)
                _canvas = color_image.enhance(factor= saturation)

            if blur_strength:
                if depth_map is not None:
                    depth_map = tensor2pil(depth_map).convert('L').convert('RGB')
                    if depth_map.size != _canvas.size:
                        depth_map.resize((_canvas.size), Image.BILINEAR)
                    _canvas = depthblur_image(_canvas, depth_map, blur_strength, focal_depth, blur_focus_spread)
                else:
                    _canvas = radialblur_image(_canvas, blur_strength, center_x, center_y, blur_focus_spread * 2)

            if vignette_intensity:
                # adjust image gamma and saturation
                _canvas = gamma_trans(_canvas, 1 - vignette_intensity / 3)
                color_image = ImageEnhance.Color(_canvas)
                _canvas = color_image.enhance(factor= 1+ vignette_intensity / 3)
                # add vignette
                _canvas = vignette_image(_canvas, vignette_intensity, center_x, center_y)

            if grain_power:
                _canvas = filmgrain_image(_canvas, grain_scale, grain_power, grain_shadows, grain_highs, grain_sat)

            ret_image = _canvas
            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
