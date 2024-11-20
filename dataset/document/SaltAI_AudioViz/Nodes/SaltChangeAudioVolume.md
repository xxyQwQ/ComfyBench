---
tags:
- Audio
---

# Audio Volume
## Documentation
- Class name: `SaltChangeAudioVolume`
- Category: `SALT/AudioViz/Audio/Process`
- Output node: `False`

The node is designed to adjust the volume of an audio file. It modifies the volume level of the input audio data based on a specified decibel value, allowing for both amplification and attenuation.
## Input types
### Required
- **`audio`**
    - The raw audio data to be processed. This parameter is crucial as it serves as the input audio file whose volume is to be adjusted.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
- **`volume_decibals`**
    - The amount, in decibels, by which to increase or decrease the volume of the input audio. This parameter directly influences the loudness of the output audio.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`audio`**
    - Comfy dtype: `AUDIO`
    - The modified audio data with the adjusted volume level.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltChangeAudioVolume:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "volume_decibals": ("FLOAT", {"min": -60, "max": 60, "default": 0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "change_volume"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"

    def change_volume(self, audio_data, volume_decibals):
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")
        modified_audio_segment = audio_segment + volume_decibals
        return (get_buffer(modified_audio_segment), )

```
