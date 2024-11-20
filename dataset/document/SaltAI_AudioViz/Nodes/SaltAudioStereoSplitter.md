---
tags:
- Audio
---

# Audio Stereo Splitter
## Documentation
- Class name: `SaltAudioStereoSplitter`
- Category: `SALT/AudioViz/Audio/Process`
- Output node: `False`

This node is designed to split a stereo audio file into two separate mono audio files, one for each channel. It ensures that audio input is in stereo format and divides it into left and right channels, providing a straightforward way to work with audio channels individually.
## Input types
### Required
- **`audio`**
    - The stereo audio input that will be split into two mono channels. It is essential for processing and must be in stereo format to ensure proper splitting.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
## Output types
- **`left_channel_mono`**
    - Comfy dtype: `AUDIO`
    - The left channel of the input stereo audio, processed and returned as a mono audio file.
    - Python dtype: `bytes`
- **`right_channel_mono`**
    - Comfy dtype: `AUDIO`
    - The right channel of the input stereo audio, processed and returned as a mono audio file.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAudioStereoSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("left_channel_mono", "right_channel_mono")
    FUNCTION = "split_stereo_to_mono"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"

    def split_stereo_to_mono(self, audio):
        stereo_audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")

        if stereo_audio_segment.channels != 2:
            raise ValueError("Input audio must be two channel stereo.")

        left_channel_mono = stereo_audio_segment.split_to_mono()[0]
        right_channel_mono = stereo_audio_segment.split_to_mono()[1]

        left_audio = get_buffer(left_channel_mono)
        right_audio = get_buffer(right_channel_mono)

        return (left_audio, right_audio)

```
