---
tags:
- Audio
---

# Save Audio
## Documentation
- Class name: `SaltSaveAudio`
- Category: `SALT/AudioViz/Audio`
- Output node: `True`

The SaltSaveAudio node is designed for saving audio content to a file in various formats. It allows specifying the filename prefix and the audio format, supporting a range of common audio file types. This functionality is essential for audio processing workflows that require the output to be stored persistently.
## Input types
### Required
- **`audio`**
    - The raw audio data to be saved. This is the primary content that the node operates on, determining the actual audio that will be written to the file.
    - Comfy dtype: `AUDIO`
    - Python dtype: `bytes`
- **`filename_prefix`**
    - A prefix for the filename under which the audio will be saved. This allows for easy identification and organization of saved audio files.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`format`**
    - The format in which the audio should be saved. This node supports saving in 'wav', 'mp3', and 'flac' formats, affecting the compatibility and quality of the saved audio file.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltSaveAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "audio_sfx"}),
                "format": (["wav", "mp3", "flac"], ),
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_audio"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio"

    def save_audio(self, audio, filename_prefix="audio_sfx", format="wav"):
        OUTPUT = folder_paths.get_output_directory()
        index = 0

        file_extension = format.lower()
        if format not in ['wav', 'mp3', 'flac']:
            logger.error(f"Unsupported format: {format}. Defaulting to WAV.")
            file_extension = "wav"
            format = "wav"

        while True:
            filename = f"{filename_prefix}_%04d.{file_extension}" % index
            full_path = os.path.realpath(os.path.join(OUTPUT, filename))
            if not os.path.exists(full_path):
                break
            index += 1

        audio_segment = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        audio_segment.export(full_path, format=format)

        logger.info(f"Audio saved to {filename} in {format.upper()} format")
        return ()

```
