---
tags:
- Audio
---

# Audio Playback Rate
## Documentation
- Class name: `SaltAudioPlaybackRate`
- Category: `SALT/AudioViz/Audio/Process`
- Output node: `False`

This node is designed to adjust the playback speed of an audio file. It allows users to increase or decrease the speed of audio playback without altering the pitch, providing a versatile tool for audio editing and manipulation.
## Input types
### Required
- **`audio`**
    - The audio input is the raw audio data that the node will process to adjust its playback speed. It serves as the primary content for speed adjustment.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
- **`speed_factor`**
    - The speed factor determines the rate at which the audio playback speed is adjusted. A value greater than 1 speeds up the playback, while a value less than 1 slows it down, offering precise control over the audio's tempo.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`audio`**
    - Comfy dtype: `AUDIO`
    - The output is the modified audio data with the adjusted playback speed. It retains the original audio's pitch while altering its tempo.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAudioPlaybackRate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "speed_factor": ("FLOAT", {"min": 0.5, "max": 12.0, "default": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "adjust_speed"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"

    def adjust_speed(cls, audio, speed_factor):
        audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        playback_rate = int(audio_segment.frame_rate * speed_factor)
        adjusted_audio_segment = audio_segment.set_frame_rate(playback_rate)
        return (get_buffer(adjusted_audio_segment),)

```
