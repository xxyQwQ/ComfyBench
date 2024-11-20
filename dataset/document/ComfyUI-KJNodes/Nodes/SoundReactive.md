---
tags:
- Audio
---

# Sound Reactive
## Documentation
- Class name: `SoundReactive`
- Category: `KJNodes/audio`
- Output node: `False`

The SoundReactive node dynamically adjusts its output based on the sound level input, allowing for real-time audio-reactive applications. It is designed to work with browser-based sound input and supports real-time diffusion processes with autoqueue functionality.
## Input types
### Required
- **`sound_level`**
    - Specifies the sound level to react to. It is a key factor in determining the node's output, adjusting the response based on the input sound level.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`start_range_hz`**
    - Defines the starting frequency range in Hertz for processing. It sets the lower bound of the frequency range of interest.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_range_hz`**
    - Sets the ending frequency range in Hertz for processing. It determines the upper limit of the frequency spectrum to be considered.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`multiplier`**
    - A multiplier applied to the sound level, allowing for amplification or attenuation of the input signal.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`smoothing_factor`**
    - Controls the smoothing of the sound level input, providing a way to reduce abrupt changes and fluctuations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`normalize`**
    - A boolean flag indicating whether the sound level should be normalized, affecting the scale of the output.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`sound_level`**
    - Comfy dtype: `FLOAT`
    - The adjusted sound level after applying the node's processing, reflecting changes based on the input and node settings.
    - Python dtype: `float`
- **`sound_level_int`**
    - Comfy dtype: `INT`
    - An integer representation of the adjusted sound level, offering a discrete version of the output for certain applications.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SoundReactive:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {  
            "sound_level": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 99999, "step": 0.01}),
            "start_range_hz": ("INT", {"default": 150, "min": 0, "max": 9999, "step": 1}),
            "end_range_hz": ("INT", {"default": 2000, "min": 0, "max": 9999, "step": 1}),
            "multiplier": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 99999, "step": 0.01}),
            "smoothing_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "normalize": ("BOOLEAN", {"default": False}),
            },
            }
    
    RETURN_TYPES = ("FLOAT","INT",)
    RETURN_NAMES =("sound_level", "sound_level_int",)
    FUNCTION = "react"
    CATEGORY = "KJNodes/audio"
    DESCRIPTION = """
Reacts to the sound level of the input.  
Uses your browsers sound input options and requires.  
Meant to be used with realtime diffusion with autoqueue.
"""
        
    def react(self, sound_level, start_range_hz, end_range_hz, smoothing_factor, multiplier, normalize):

        sound_level *= multiplier

        if normalize:
            sound_level /= 255

        sound_level_int = int(sound_level)
        return (sound_level, sound_level_int, )     

```
