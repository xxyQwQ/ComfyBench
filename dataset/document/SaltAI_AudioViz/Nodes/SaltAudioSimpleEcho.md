---
tags:
- Audio
---

# Audio Simple Echo
## Documentation
- Class name: `SaltAudioSimpleEcho`
- Category: `SALT/AudioViz/Audio/Effect`
- Output node: `False`

The SaltAudioSimpleEcho node applies an echo effect to audio inputs, allowing for the creation of layered audio experiences by repeating the audio signal multiple times with a specified delay and decay factor.
## Input types
### Required
- **`audio`**
    - The raw audio data to which the echo effect will be applied. This is the primary input for the echo processing.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
- **`times`**
    - Specifies the number of times the audio will be echoed, affecting the density of the echo effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`delay_ms`**
    - The delay in milliseconds between each echo, determining the temporal spacing of the echoes.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`decay_factor`**
    - The factor by which the volume of each successive echo decreases, controlling the fade-out effect of the echoes.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`audio`**
    - Comfy dtype: `AUDIO`
    - The processed audio data with the echo effect applied.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAudioSimpleEcho:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "times": ("INT", {"default": 3, "min": 1, "max": 10}),
                "delay_ms": ("INT", {"default": 100, "min": 100, "max": 2000}),
                "decay_factor": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 0.9, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "apply_echo"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    def apply_echo(self, audio, times, delay_ms, decay_factor):
        original = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        silence = AudioSegment.silent(duration=delay_ms)
        echo = original
        fade_duration = int(delay_ms * 0.1)
        speed_increase_factor = 1.01
        speed_increase_step = 0.01

        for i in range(1, times):
            decayed = original - (decay_factor * 10 * i)
            playback_speed = speed_increase_factor + (speed_increase_step * i)
            decayed = decayed.speedup(playback_speed=playback_speed)
            for j in range(1, 5):
                decayed = decayed.overlay(silence + decayed - (5 * j), position=50 * j)
            decayed_with_fades = decayed.fade_in(fade_duration).fade_out(fade_duration)
            echo = echo.overlay(silence + decayed_with_fades, position=delay_ms * i)

        return (get_buffer(echo),)

```
