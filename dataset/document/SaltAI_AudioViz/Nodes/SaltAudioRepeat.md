---
tags:
- Audio
---

# Audio Repeat
## Documentation
- Class name: `SaltAudioRepeat`
- Category: `SALT/AudioViz/Audio/Process`
- Output node: `False`

The SaltAudioRepeat node is designed for repeating an audio segment a specified number of times. It enables the creation of looped audio effects by duplicating the input audio according to the user-defined repetition count.
## Input types
### Required
- **`audio`**
    - The 'audio' parameter is the raw audio data to be looped. It is essential for determining the base audio content that will be repeated.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
- **`repeat_times`**
    - The 'repeat_times' parameter specifies how many times the audio should be looped. It directly influences the length and content of the output audio.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`audio`**
    - Comfy dtype: `AUDIO`
    - The output is the looped audio, which is the result of repeating the input audio segment for the specified number of times.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAudioRepeat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "repeat_times": ("INT", {"min": 1, "default": 2}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "loop_audio"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"

    def loop_audio(cls, audio, repeat_times):
        audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        looped_audio_segment = audio_segment * repeat_times
        return (get_buffer(looped_audio_segment),)

```
