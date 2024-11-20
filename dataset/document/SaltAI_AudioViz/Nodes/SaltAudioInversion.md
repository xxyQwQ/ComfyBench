---
tags:
- Audio
---

# Audio Reverse
## Documentation
- Class name: `SaltAudioInversion`
- Category: `SALT/AudioViz/Audio/Process`
- Output node: `False`

The node is designed to invert the waveform of an audio file, effectively flipping its phase. This process can be used to create unique sound effects or to cancel out specific audio components when mixed with the original.
## Input types
### Required
- **`audio`**
    - The audio input is the raw audio data that will be inverted. This process alters the audio's waveform by reversing its phase, which can be useful for sound design or audio correction purposes.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
## Output types
- **`audio`**
    - Comfy dtype: `AUDIO`
    - The output is the inverted audio data, with its waveform phase reversed from the original input. This can be used for creative audio effects or technical audio manipulation.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAudioInversion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "invert_audio"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"

    def invert_audio(cls, audio):
        audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        inverted_samples = np.array(audio_segment.get_array_of_samples()) * -1
        inverted_audio_segment = audio_segment._spawn(inverted_samples.tobytes())
        return (get_buffer(inverted_audio_segment),)

```
