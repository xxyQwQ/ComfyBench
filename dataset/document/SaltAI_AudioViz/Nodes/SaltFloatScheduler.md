---
tags:
- Scheduling
---

# Float Schedule
## Documentation
- Class name: `SaltFloatScheduler`
- Category: `SALT/AudioViz/Scheduling`
- Output node: `False`

The SaltFloatScheduler node is designed for creating and managing float schedules, which are sequences of floating-point numbers that can be used to control various parameters over time. This node likely offers functionality to define, adjust, and iterate over these float schedules, providing a flexible tool for temporal parameter manipulation in audio-visual projects.
## Input types
### Required
- **`repeat_sequence_times`**
    - Specifies the number of times the sequence should be repeated, affecting the overall length and repetition pattern of the float schedule.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`curves_mode`**
    - Determines the mode of curve application for the float schedule, influencing how the values interpolate or transition between points in the schedule.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`use_perlin_tremors`**
    - Indicates whether Perlin noise-based tremors should be applied to the float schedule, adding a layer of complexity and variability to the schedule's progression.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`tremor_scale`**
    - Sets the scale of the Perlin tremors, affecting the granularity and frequency of the noise applied to the float schedule.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`tremor_octaves`**
    - Defines the number of octaves for the Perlin tremors, influencing the detail and layering of the noise effect on the float schedule.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`tremor_persistence`**
    - Controls the persistence of the Perlin tremors, determining the amplitude's decay rate across octaves in the float schedule.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`tremor_lacunarity`**
    - Specifies the lacunarity of the Perlin tremors, affecting the frequency growth per octave in the float schedule.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sequence`**
    - The sequence of values to be scheduled, serving as the base for the float schedule before any modifications or enhancements.
    - Comfy dtype: `STRING`
    - Python dtype: `List[float]`
### Optional
- **`max_sequence_length`**
    - Limits the maximum length of the float schedule, ensuring that the sequence does not exceed a specified number of elements.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`schedule_list`**
    - Comfy dtype: `LIST`
    - The resulting float schedule after applying the specified parameters and modifications, ready for use in controlling parameters over time.
    - Python dtype: `List[float]`
- **`schedule_length`**
    - Comfy dtype: `INT`
    - The length of the generated float schedule, providing information on the total number of elements in the schedule.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltFloatScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        easing_fns = list(easing_functions.keys())
        easing_fns.insert(0, "None")
        return {
            "required": {
                "repeat_sequence_times": ("INT", {"default": 0, "min": 0}),
                "curves_mode": (easing_fns, ),
                "use_perlin_tremors": ("BOOLEAN", {"default": True}),
                "tremor_scale": ("FLOAT", {"default": 64, "min": 0.01, "max": 1024.0, "step": 0.01}),
                "tremor_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "tremor_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "tremor_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 3.0, "step": 0.01}),
                "sequence": ("STRING", {"multiline": True, "placeholder": "[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]"}),
            },
            "optional": {
                "max_sequence_length": ("INT", {"default": 0, "min": 0, "max": 4096}),
            }
        }

    RETURN_TYPES = ("LIST", "INT")
    RETURN_NAMES = ("schedule_list", "schedule_length")
    FUNCTION = "generate_sequence"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling"

    def apply_curve(self, sequence, mode):
        if mode in easing_functions.keys():
            sequence = [easing_functions[mode](t) for t in sequence]
        else:
            logger.error(f"The easing mode `{mode}` does not exist in the valid easing functions: {', '.join(easing_functions.keys())}")
        return sequence

    def apply_perlin_noise(self, sequence, scale, octaves, persistence, lacunarity):
        perlin = PerlinNoise()
        noise_values = [perlin.sample(i, scale=scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity) for i, _ in enumerate(sequence)]
        sequence = [val + noise for val, noise in zip(sequence, noise_values)]
        return sequence

    def generate_sequence(self, sequence, repeat_sequence_times, curves_mode, use_perlin_tremors, tremor_scale, tremor_octaves, tremor_persistence, tremor_lacunarity, max_sequence_length=0):
        sequence_list = [float(val.strip()) for val in sequence.replace("[", "").replace("]", "").split(',')]
        if use_perlin_tremors:
            sequence_list = self.apply_perlin_noise(sequence_list, tremor_scale, tremor_octaves, tremor_persistence, tremor_lacunarity)
        if curves_mode != "None":
            sequence_list = self.apply_curve(sequence_list, curves_mode)
        sequence_list = sequence_list * (repeat_sequence_times + 1)
        sequence_list = sequence_list[:max_sequence_length] if max_sequence_length != 0 else sequence_list
        return (sequence_list, len(sequence_list))

```
