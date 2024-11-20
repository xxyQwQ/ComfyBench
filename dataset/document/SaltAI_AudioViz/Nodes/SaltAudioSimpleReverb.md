---
tags:
- Audio
---

# Audio Simple Reverb
## Documentation
- Class name: `SaltAudioSimpleReverb`
- Category: `SALT/AudioViz/Audio/Effect`
- Output node: `False`

This node applies a simple reverb effect to an audio input, simulating the sound reflection and decay characteristics of a physical space. It allows for the adjustment of the reverb level and decay time to tailor the effect.
## Input types
### Required
- **`audio`**
    - The raw audio data to which the reverb effect will be applied. This input is crucial for defining the base sound that will be processed.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
- **`reverb_level`**
    - Specifies the intensity of the reverb effect. A higher value results in a more pronounced reverb, simulating a larger or more reflective space.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`decay`**
    - Determines the rate at which the reverb effect fades away. A higher decay value simulates a longer echo, as if the sound is reflecting off distant surfaces.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`audio`**
    - Comfy dtype: `AUDIO`
    - The processed audio data with the reverb effect applied, reflecting the adjustments made through the input parameters.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAudioSimpleReverb:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "reverb_level": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01}),
                "decay": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "apply_reverb"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    def apply_reverb(self, audio, reverb_level, decay):
        original = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        reverb_duration = int(len(original) * reverb_level)
        output = original

        for overlay_delay in range(50, reverb_duration, 50):
            decayed_overlay = original - (decay * overlay_delay)
            output = output.overlay(decayed_overlay, position=overlay_delay)

        return (get_buffer(output),)

```
