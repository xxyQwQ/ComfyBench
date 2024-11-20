---
tags:
- Audio
---

# Audio Stitcher
## Documentation
- Class name: `SaltAudioStitcher`
- Category: `SALT/AudioViz/Audio/Process`
- Output node: `False`

The SaltAudioStitcher node is designed to seamlessly combine two audio files into a single audio track. It allows for the insertion of a silent transition between the two audio segments, providing flexibility in creating continuous audio experiences from separate sources.
## Input types
### Required
- **`audio_a`**
    - The first audio file to be stitched. It serves as the starting point of the combined audio track.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
- **`audio_b`**
    - The second audio file to be stitched. It follows the first audio file in the combined audio track.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
### Optional
- **`silent_transition_seconds`**
    - The duration of silence to insert between the two audio files, allowing for a customizable pause or transition in the combined audio track.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`stitched_audio`**
    - Comfy dtype: `AUDIO`
    - The resulting audio file after stitching the two input audio files, optionally separated by a silent transition.
    - Python dtype: `bytes`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAudioStitcher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_a": ("AUDIO", {}),
                "audio_b": ("AUDIO", {}),
            },
            "optional": {
                "silent_transition_seconds": ("FLOAT", {"min": 0.0, "default": 0.0}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("stitched_audio",)
    FUNCTION = "stitch_audios"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/Process"

    def stitch_audios(self, audio_a, audio_b, silent_transition_seconds=0.0):
        audio_segment_1 = AudioSegment.from_file(io.BytesIO(audio_a), format="wav")
        audio_segment_2 = AudioSegment.from_file(io.BytesIO(audio_b), format="wav")

        if silent_transition_seconds > 0:
            silence_segment = AudioSegment.silent(duration=int(silent_transition_seconds * 1000))
            stitched = audio_segment_1 + silence_segment + audio_segment_2
        else:
            stitched = audio_segment_1 + audio_segment_2

        return (get_buffer(stitched), )

```
