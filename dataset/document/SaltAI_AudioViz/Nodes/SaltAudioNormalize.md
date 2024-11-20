---
tags:
- Audio
---

# Audio Normalize
## Documentation
- Class name: `SaltAudioNormalize`
- Category: `SALT/AudioViz/Audio/Process`
- Output node: `False`

The SaltAudioNormalize node is designed to normalize audio files, adjusting their volume levels to a consistent amplitude. This process enhances the audio quality by ensuring that the sound is neither too loud nor too soft, making it ideal for preparing audio for further processing or playback.
## Input types
### Required
- **`audio`**
    - The 'audio' input is the raw audio data that needs normalization. It is crucial for the node's operation as it directly influences the outcome of the normalization process.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
## Output types
- **`audio`**
    - Comfy dtype: `AUDIO`
    - The 'audio' output is the normalized version of the input audio. It represents the result of the normalization process, with adjusted volume levels for consistent amplitude.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAudioNormalize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "normalize_audio"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"

    def normalize_audio(self, audio):
        audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        normalized_segment = AudioEffects.normalize(audio_segment)
        return (get_buffer(normalized_segment), )

```
