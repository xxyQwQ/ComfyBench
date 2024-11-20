---
tags:
- Audio
---

# Audio Stereo Merge
## Documentation
- Class name: `SaltAudioStereoMerge`
- Category: `SALT/AudioViz/Audio/Effect`
- Output node: `False`

This node merges two mono audio inputs into a single stereo audio output, ensuring that each input occupies one channel of the stereo output. It is designed to work with audio files in WAV format, automatically adjusting the channels of the inputs to mono if necessary and aligning their lengths before merging.
## Input types
### Required
- **`audio_a`**
    - The first mono audio input to be merged into the stereo output. It occupies the left channel of the resulting stereo audio.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
- **`audio_b`**
    - The second mono audio input to be merged into the stereo output. It occupies the right channel of the resulting stereo audio.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
## Output types
- **`audio`**
    - Comfy dtype: `AUDIO`
    - The resulting stereo audio output, with the first input occupying the left channel and the second input occupying the right channel.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAudioStereoMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_a": ("AUDIO", {}),
                "audio_b": ("AUDIO", {}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "merge_stereo"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Effect"

    def merge_stereo(self, audio_a, audio_b):
        segment_a = AudioSegment.from_file(io.BytesIO(audio_a), format="wav")
        segment_b = AudioSegment.from_file(io.BytesIO(audio_b), format="wav")
        
        if segment_a.channels > 1:
            segment_a = segment_a.set_channels(1)
        if segment_b.channels > 1:
            segment_b = segment_b.set_channels(1)
        
        min_length = min(len(segment_a), len(segment_b))
        segment_a = segment_a[:min_length]
        segment_b = segment_b[:min_length]
        stereo_audio = AudioSegment.from_mono_audiosegments(segment_a, segment_b)
        
        stereo_audio_bytes = io.BytesIO()
        stereo_audio.export(stereo_audio_bytes, format="wav")
        stereo_audio_bytes.seek(0)
        
        return stereo_audio_bytes.read(),

```
