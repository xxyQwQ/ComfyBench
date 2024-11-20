---
tags:
- Audio
---

# Audio Mixer
## Documentation
- Class name: `SaltAudioMixer`
- Category: `SALT/AudioViz/Audio/Process`
- Output node: `False`

The SaltAudioMixer node is designed for blending two audio streams into a single output by overlaying one audio stream onto another at a specified mix time. This functionality is essential for creating composite audio tracks or for audio editing purposes where multiple sounds need to be combined.
## Input types
### Required
- **`audio_a`**
    - The first audio stream to be mixed. It serves as the base layer onto which the second audio stream will be overlaid.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
- **`audio_b`**
    - The second audio stream to be mixed. It is overlaid onto the first audio stream at the specified mix time.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
- **`mix_time_seconds`**
    - The time, in seconds, at which the second audio stream is to be overlaid onto the first. This determines the starting point of the overlay in the resulting mixed audio.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`mixed_audio`**
    - Comfy dtype: `AUDIO`
    - The resulting audio stream after mixing the two input streams. It represents the composite audio track created by overlaying the second audio stream onto the first at the specified mix time.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAudioMixer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_a": ("AUDIO", {}),
                "audio_b": ("AUDIO", {}),
                "mix_time_seconds": ("FLOAT", {"min": 0.0, "default": 0.0}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("mixed_audio",)
    FUNCTION = "mix_audios"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"

    def mix_audios(self, audio_a, audio_b, mix_time_seconds):
        audio_segment_1 = AudioSegment.from_file(io.BytesIO(audio_a), format="wav")
        audio_segment_2 = AudioSegment.from_file(io.BytesIO(audio_b), format="wav")

        position_ms = int(mix_time_seconds * 1000)
        audio_segment = audio_segment_1.overlay(audio_segment_2, position=position_ms)

        return (get_buffer(audio_segment), )

```
