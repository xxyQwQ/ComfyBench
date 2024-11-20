---
tags:
- Multimedia
- VideoHelperSuite
---

# Load Audio (Path)🎥🅥🅗🅢
## Documentation
- Class name: `VHS_LoadAudio`
- Category: `Video Helper Suite 🎥🅥🅗🅢`
- Output node: `False`

The VHS_LoadAudio node is designed for loading audio files within the Video Helper Suite, allowing users to input audio files from specified paths and optionally seek to a certain point within the audio file. This functionality is crucial for tasks that involve audio processing or synchronization with video content, providing a foundational tool for audio manipulation in multimedia projects.
## Input types
### Required
- **`audio_file`**
    - Specifies the path to the audio file to be loaded. The node supports a variety of audio formats, including wav, mp3, ogg, m4a, and flac, making it versatile for different audio processing needs.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`seek_seconds`**
    - Defines the number of seconds to seek into the audio file before starting to load, allowing for precise control over the portion of the audio that is processed. This is useful for skipping unwanted content at the beginning of an audio file or for focusing on a specific segment.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`audio`**
    - Comfy dtype: `VHS_AUDIO`
    - Returns a handle to the loaded audio, enabling further processing or analysis. This output is essential for integrating audio content into multimedia applications or workflows.
    - Python dtype: `function`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LoadAudio:
    @classmethod
    def INPUT_TYPES(s):
        #Hide ffmpeg formats if ffmpeg isn't available
        return {
            "required": {
                "audio_file": ("STRING", {"default": "input/", "vhs_path_extensions": ['wav','mp3','ogg','m4a','flac']}),
                },
            "optional" : {"seek_seconds": ("FLOAT", {"default": 0, "min": 0})}
        }

    RETURN_TYPES = ("VHS_AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "load_audio"
    def load_audio(self, audio_file, seek_seconds):
        audio_file = strip_path(audio_file)
        if audio_file is None or validate_path(audio_file) != True:
            raise Exception("audio_file is not a valid path: " + audio_file)
        #Eagerly fetch the audio since the user must be using it if the
        #node executes, unlike Load Video
        audio = get_audio(audio_file, start_time=seek_seconds)
        return (lambda : audio,)

    @classmethod
    def IS_CHANGED(s, audio_file, seek_seconds):
        return hash_path(audio_file)

    @classmethod
    def VALIDATE_INPUTS(s, audio_file, **kwargs):
        return validate_path(audio_file, allow_none=True)

```
