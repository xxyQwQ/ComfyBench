---
tags:
- Audio
---

# Audio Fade
## Documentation
- Class name: `SaltAudioFade`
- Category: `SALT/AudioViz/Audio/Process`
- Output node: `False`

The SaltAudioFade node applies a fade-in or fade-out effect to an audio clip, based on specified parameters such as fade type, duration, and start time. This node is designed to manipulate the dynamics of an audio segment, enhancing its auditory transitions.
## Input types
### Required
- **`audio`**
    - The raw audio data to which the fade effect will be applied. This is the primary input for the fade operation.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
- **`fade_type`**
    - Specifies the type of fade effect to apply: 'in' for a fade-in effect, or 'out' for a fade-out effect. This determines the direction of the volume transition.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`fade_duration`**
    - The duration of the fade effect in seconds. This defines how long the fade transition will last.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`fade_start`**
    - The start time in seconds for the fade effect. This optional parameter allows for specifying when the fade effect should begin within the audio clip.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`audio`**
    - Comfy dtype: `AUDIO`
    - The modified audio data with the applied fade effect. This output is suitable for further processing or playback.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAudioFade:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "fade_type": (["in", "out"],),
                "fade_duration": ("FLOAT", {"min": 0.01}),
            },
            "optional": {
                "fade_start": ("FLOAT", {"default": 0, "min": 0}),
            },
        }

    RETURN_TYPES = ("AUDIO", )
    RETURN_NAMES = ("audio", )
    FUNCTION = "apply_fade"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"


    def apply_fade(self, audio, fade_type, fade_duration, fade_start=0):
        audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")

        start_ms = int(fade_start * 1000)
        duration_ms = int(fade_duration * 1000)
        end_fade_ms = start_ms + duration_ms

        before_fade = audio_segment[:start_ms]
        during_fade = audio_segment[start_ms:end_fade_ms]
        after_fade = audio_segment[end_fade_ms:]

        if fade_type == "in":
            faded_part = during_fade.fade_in(duration_ms)
        else:
            faded_part = during_fade.fade_out(duration_ms)

        output = before_fade + faded_part + after_fade

        return (get_buffer(output), )

```
