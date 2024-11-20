---
tags:
- Multimedia
- VideoHelperSuite
---

# Load Audio (Upload)🎥🅥🅗🅢
## Documentation
- Class name: `VHS_LoadAudioUpload`
- Category: `Video Helper Suite 🎥🅥🅗🅢`
- Output node: `False`

The VHS_LoadAudioUpload node is designed for uploading and processing audio files within the Video Helper Suite. It allows users to upload audio files, specifying the desired starting point and duration for processing. This node is essential for integrating audio content into video projects, enabling precise control over the audio segment to be used.
## Input types
### Required
- **`audio`**
    - Specifies the audio file to be uploaded from a predefined list of available files. This selection is crucial for determining the specific audio content to be processed.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`start_time`**
    - Determines the starting point, in seconds, from which the audio file should be processed. This parameter allows for selective use of audio content within a larger file.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`duration`**
    - Specifies the duration, in seconds, for which the audio from the starting point should be processed. This enables precise control over the segment of the audio file to be used.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`audio`**
    - Comfy dtype: `VHS_AUDIO`
    - Returns the processed audio segment, allowing it to be integrated into video projects or further manipulated.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LoadAudioUpload:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1] in audio_extensions):
                    files.append(f)
        return {"required": {
                    "audio": (sorted(files),),
                    "start_time": ("FLOAT" , {"default": 0, "min": 0, "max": 10000000, "step": 0.01}),
                    "duration": ("FLOAT" , {"default": 0, "min": 0, "max": 10000000, "step": 0.01}),
                     },
                }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = ("VHS_AUDIO", )
    RETURN_NAMES = ("audio",)
    FUNCTION = "load_audio"

    def load_audio(self, start_time, duration, **kwargs):
        audio_file = folder_paths.get_annotated_filepath(strip_path(kwargs['audio']))
        if audio_file is None or validate_path(audio_file) != True:
            raise Exception("audio_file is not a valid path: " + audio_file)
        
        audio = get_audio(audio_file, start_time, duration)

        return (lambda : audio,)

    @classmethod
    def IS_CHANGED(s, audio, start_time, duration):
        audio_file = folder_paths.get_annotated_filepath(strip_path(audio))
        return hash_path(audio_file)

    @classmethod
    def VALIDATE_INPUTS(s, audio, **kwargs):
        audio_file = folder_paths.get_annotated_filepath(strip_path(audio))
        return validate_path(audio_file, allow_none=True)

```
