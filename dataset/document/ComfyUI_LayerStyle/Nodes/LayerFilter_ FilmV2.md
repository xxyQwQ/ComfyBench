---
tags:
- VisualEffects
---

# LayerFilter: Film V2
## Documentation
- Class name: `LayerFilter: FilmV2`
- Category: `😺dzNodes/LayerFilter`
- Output node: `False`

The FilmV2 node applies a sophisticated film layer filter to images, enhancing them with a cinematic quality. It leverages advanced image processing techniques to simulate the aesthetic and characteristics of film photography, offering users a unique way to stylize their visuals.
## Input types
### Required
- **`image`**
    - The 'image' input type represents the image data to which the FilmV2 filter will be applied. It is crucial for defining the visual content that will undergo the transformation, serving as the foundation for the filter's effects.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`center_x`**
    - Specifies the horizontal center of the effect focus area, influencing how certain effects like blur are centered on the image. It plays a key role in the placement of visual enhancements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`center_y`**
    - Defines the vertical center of the effect focus area, affecting the positioning of effects such as blur. Essential for tailoring the filter's impact to specific areas of the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`saturation`**
    - Controls the intensity of the image's colors. Adjusting saturation can dramatically alter the mood and visual style of the output, making it a key parameter for customization.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`vignette_intensity`**
    - Determines the strength of the vignette effect applied to the edges of the image, contributing to the cinematic feel by darkening the periphery.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`grain_method`**
    - Selects the algorithm used to add grain to the image, offering options to customize the texture and appearance of the film effect. The choice between 'fastgrain' and 'filmgrainer' methods affects the texture and visual quality of the grain, with each method providing a distinct style and impact on the final image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`grain_power`**
    - Adjusts the intensity of the grain effect, allowing for fine-tuning the level of perceived film grain and texture.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`grain_scale`**
    - Modifies the scale of the grain particles, affecting their size and visibility on the image. This parameter helps in achieving the desired level of detail and texture.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`grain_sat`**
    - Sets the saturation level of the grain effect, influencing how colorfully the grain appears on the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`filmgrainer_shadows`**
    - Controls the visibility of grain in the darker areas of the image, enabling more precise customization of the film effect in shadows.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`filmgrainer_highs`**
    - Adjusts the grain effect in the highlights, allowing for nuanced control over the appearance of grain in brighter image areas.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`blur_strength`**
    - Determines the intensity of the blur effect, which can be used to simulate depth of field or focus effects.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`blur_focus_spread`**
    - Controls the spread of the blur effect around the focus point, affecting how gradually the image transitions from sharp to blurred.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`focal_depth`**
    - Sets the depth at which the focus effect is centered, crucial for achieving realistic depth of field simulations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`depth_map`**
    - An optional input that provides a depth map for more advanced depth-based effects, such as variable blur based on distance from the camera. Including a depth map enables the node to apply blur effects more accurately by simulating the depth of field based on the map, thus enhancing the realism of the film effect.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output 'image' type is the transformed version of the input image, now bearing the distinctive qualities of the FilmV2 filter. It signifies the culmination of the filter's application, showcasing the enhanced cinematic effect.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class FilmV2:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        grain_method_list = ["fastgrain", "filmgrainer", ]
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "center_x": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1, "min": 0.01, "max": 3, "step": 0.01}),
                "vignette_intensity": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "grain_method": (grain_method_list,),
                "grain_power": ("FLOAT", {"default": 0.15, "min": 0, "max": 1, "step": 0.01}),
                "grain_scale": ("FLOAT", {"default": 1, "min": 0.1, "max": 10, "step": 0.1}),
                "grain_sat": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "filmgrainer_shadows": ("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step": 0.01}),
                "filmgrainer_highs": ("FLOAT", {"default": 0.2, "min": 0, "max": 1, "step": 0.01}),
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
    FUNCTION = 'film_v2'
    CATEGORY = '😺dzNodes/LayerFilter'

    def film_v2(self, image, center_x, center_y, saturation, vignette_intensity,
                  grain_method, grain_power, grain_scale, grain_sat, filmgrainer_shadows, filmgrainer_highs,
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
                    depth_map = tensor2pil(depth_map).convert('RGB')
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
                if grain_method == "fastgrain":
                    _canvas = image_add_grain(_canvas, grain_scale,grain_power, grain_sat, toe=0, seed=int(time.time()))
                elif grain_method == "filmgrainer":
                    _canvas = filmgrain_image(_canvas, grain_scale, grain_power, filmgrainer_shadows, filmgrainer_highs, grain_sat)

            ret_image = _canvas
            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
