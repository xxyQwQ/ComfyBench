---
tags:
- Scheduling
---

# Apply Easing to Schedule
## Documentation
- Class name: `SaltScheduleVariance`
- Category: `SALT/AudioViz/Scheduling`
- Output node: `False`

This node applies a combination of noise-based tremors and easing functions to a schedule list to create varied animation frames. It allows for the dynamic adjustment of animation parameters over time, enhancing visual effects with perlin noise and optional curve adjustments.
## Input types
### Required
- **`schedule_list`**
    - The list of values representing the schedule to be varied. It serves as the base for applying noise and curve adjustments, directly influencing the visual dynamics of the animation.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
### Optional
- **`curves_mode`**
    - Specifies the type of easing function to apply to the schedule list, offering a method to smoothly transition between values.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`use_perlin_tremors`**
    - A flag to enable or disable the application of perlin noise-based tremors to the schedule list, adding a layer of dynamic variation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`tremor_scale`**
    - Determines the intensity of the noise-based tremors applied to the schedule list, affecting the degree of variation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`octaves`**
    - Controls the number of layers of noise to combine for generating tremors, impacting the complexity of the resulting variation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`persistence`**
    - Affects the amplitude of each octave in the noise function, influencing the smoothness of the tremors.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`lacunarity`**
    - Determines the frequency of each octave in the noise function, affecting the detail of the tremors.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`schedule_list`**
    - Comfy dtype: `LIST`
    - The modified schedule list after applying noise-based tremors and optional curve adjustments, ready for use in animating visual elements.
    - Python dtype: `List[float]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltScheduleVariance:
    """
    Applies Perlin noise and optional easing curves to each value in a list to create an OPAC Schedule out of it,
    while aiming to preserve the original distribution of the input values.
    """
    def __init__(self):
        self.noise_base = random.randint(0, 1000)
        self.perlin_noise = PerlinNoise()

    @classmethod
    def INPUT_TYPES(cls):
        easing_fn = list(easing_functions.keys())
        easing_fn.insert(0, "None")
        return {
            "required": {
                "schedule_list": ("LIST", {}),
            },
            "optional": {
                "curves_mode": (easing_fn,),
                "use_perlin_tremors": ("BOOLEAN", {"default": True}),
                "tremor_scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 1.0}),
                "octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0}),
                "lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("schedule_list",)
    FUNCTION = "opac_variance"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling"

    def sample_perlin(self, frame_index, value, tremor_scale, octaves, persistence, lacunarity):
        noise = self.perlin_noise.sample(self.noise_base + frame_index * 0.1, scale=1.0, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
        noise_adjustment = 1 + noise * tremor_scale
        return max(0, min(value * noise_adjustment, 1))

    def opac_variance(self, schedule_list, curves_mode="None", use_perlin_tremors=True, tremor_scale=0.05, octaves=1, persistence=0.5, lacunarity=2.0):
        self.frame_count = len(schedule_list)
        varied_list = schedule_list.copy()

        if use_perlin_tremors:
            for i, value in enumerate(schedule_list):
                noise_adjusted_value = self.sample_perlin(i, value, tremor_scale, octaves, persistence, lacunarity)
                varied_list[i] = round(noise_adjusted_value, 2)

        if curves_mode != "None" and curves_mode in easing_functions:
            for i, value in enumerate(varied_list):
                curve_adjustment = easing_functions[curves_mode](i / max(1, (self.frame_count - 1)))
                # Apply curve adjustment to the original value, not to the noise-adjusted value
                original_value_with_curve = curve_adjustment * schedule_list[i]
                # Blend the original value with curves and noise-adjusted value
                varied_list[i] = round((value + original_value_with_curve) / 2, 2)

        return (varied_list,)

```
