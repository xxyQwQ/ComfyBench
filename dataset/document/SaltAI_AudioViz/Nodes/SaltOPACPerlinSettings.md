---
tags:
- Scheduling
- VisualEffects
---

# Perlin Tremor Settings
## Documentation
- Class name: `SaltOPACPerlinSettings`
- Category: `SALT/AudioViz/Scheduling`
- Output node: `False`

This node is designed for configuring Perlin noise parameters specifically for the OPAC visualization node. It allows for the dynamic adjustment of Perlin noise settings to influence visual effects based on audio or other inputs, enhancing the visual representation of data with more natural, organic variations.
## Input types
### Required
- **`zoom_octaves`**
    - Specifies the number of octaves for zoom-related Perlin noise, affecting the level of detail in the zoom effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`zoom_persistence`**
    - Determines the persistence of zoom-related Perlin noise, influencing the smoothness of the zoom effect in the visualization.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`zoom_lacunarity`**
    - Controls the lacunarity of zoom-related Perlin noise, affecting the granularity of the zoom effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`zoom_repeat`**
    - Specifies the repeat interval for zoom-related Perlin noise, defining how often the noise pattern repeats.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`angle_octaves`**
    - Specifies the number of octaves for angle-related Perlin noise, affecting the level of detail in the angle effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`angle_persistence`**
    - Determines the persistence of angle-related Perlin noise, influencing the smoothness of the angle effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`angle_lacunarity`**
    - Controls the lacunarity of angle-related Perlin noise, affecting the granularity of the angle effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`angle_repeat`**
    - Specifies the repeat interval for angle-related Perlin noise, defining how often the noise pattern repeats.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`trx_octaves`**
    - Specifies the number of octaves for translation-related Perlin noise along the x-axis, affecting the level of detail in the translation effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`trx_persistence`**
    - Determines the persistence of translation-related Perlin noise along the x-axis, influencing the smoothness of the translation effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`trx_lacunarity`**
    - Controls the lacunarity of translation-related Perlin noise along the x-axis, affecting the granularity of the translation effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`trx_repeat`**
    - Specifies the repeat interval for translation-related Perlin noise along the x-axis, defining how often the noise pattern repeats.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`try_octaves`**
    - Specifies the number of octaves for translation-related Perlin noise along the y-axis, affecting the level of detail in the translation effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`try_persistence`**
    - Determines the persistence of translation-related Perlin noise along the y-axis, influencing the smoothness of the translation effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`try_lacunarity`**
    - Controls the lacunarity of translation-related Perlin noise along the y-axis, affecting the granularity of the translation effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`try_repeat`**
    - Specifies the repeat interval for translation-related Perlin noise along the y-axis, defining how often the noise pattern repeats.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`trz_octaves`**
    - Specifies the number of octaves for translation-related Perlin noise along the z-axis, affecting the level of detail in the translation effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`trz_persistence`**
    - Determines the persistence of translation-related Perlin noise along the z-axis, influencing the smoothness of the translation effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`trz_lacunarity`**
    - Controls the lacunarity of translation-related Perlin noise along the z-axis, affecting the granularity of the translation effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`trz_repeat`**
    - Specifies the repeat interval for translation-related Perlin noise along the z-axis, defining how often the noise pattern repeats.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`rotx_octaves`**
    - Specifies the number of octaves for rotation-related Perlin noise around the x-axis, affecting the level of detail in the rotation effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`rotx_persistence`**
    - Determines the persistence of rotation-related Perlin noise around the x-axis, influencing the smoothness of the rotation effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`rotx_lacunarity`**
    - Controls the lacunarity of rotation-related Perlin noise around the x-axis, affecting the granularity of the rotation effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`rotx_repeat`**
    - Specifies the repeat interval for rotation-related Perlin noise around the x-axis, defining how often the noise pattern repeats.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`roty_octaves`**
    - Specifies the number of octaves for rotation-related Perlin noise around the y-axis, affecting the level of detail in the rotation effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`roty_persistence`**
    - Determines the persistence of rotation-related Perlin noise around the y-axis, influencing the smoothness of the rotation effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`roty_lacunarity`**
    - Controls the lacunarity of rotation-related Perlin noise around the y-axis, affecting the granularity of the rotation effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`roty_repeat`**
    - Specifies the repeat interval for rotation-related Perlin noise around the y-axis, defining how often the noise pattern repeats.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`rotz_octaves`**
    - Specifies the number of octaves for rotation-related Perlin noise around the z-axis, affecting the level of detail in the rotation effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`rotz_persistence`**
    - Determines the persistence of rotation-related Perlin noise around the z-axis, influencing the smoothness of the rotation effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`rotz_lacunarity`**
    - Controls the lacunarity of rotation-related Perlin noise around the z-axis, affecting the granularity of the rotation effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`rotz_repeat`**
    - Specifies the repeat interval for rotation-related Perlin noise around the z-axis, defining how often the noise pattern repeats.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`opac_perlin_settings`**
    - Comfy dtype: `DICT`
    - The modified schedule list after applying Perlin noise and optional easing curves, ready for use in OPAC visualization.
    - Python dtype: `List[float]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltOPACPerlinSettings:
    """
        Configuration node for Perlin noise sampling in OPAC node
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zoom_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "zoom_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "zoom_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "zoom_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "angle_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "angle_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "angle_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "angle_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "trx_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "trx_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "trx_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "trx_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "try_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "try_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "try_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "try_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "trz_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "trz_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "trz_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "trz_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "rotx_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "rotx_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "rotx_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "rotx_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "roty_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "roty_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "roty_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "roty_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "rotz_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "rotz_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "rotz_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
                "rotz_repeat": ("INT", {"default": 1024, "min": 256, "max": 4096}),
            }
        }
    
    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("opac_perlin_settings",)
    FUNCTION = "process"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling"

    def process(self, **kwargs):
        return (kwargs, )

```
