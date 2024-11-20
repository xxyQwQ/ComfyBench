---
tags:
- Audio
---

# LoadAudio
## Documentation
- Class name: `LoadAudio`
- Category: `_for_testing/audio`
- Output node: `False`

The LoadAudio node is designed for loading audio files from a specified directory, supporting a range of audio formats. It facilitates the process of reading audio data into a format suitable for further audio processing or analysis tasks.
## Input types
### Required
- **`audio`**
    - Specifies the audio file to be loaded. The choice of file affects the waveform and sample rate obtained, which are crucial for any subsequent audio processing or analysis.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`audio`**
    - Comfy dtype: `AUDIO`
    - Returns a dictionary containing the waveform and sample rate of the loaded audio file, ready for further processing or analysis.
    - Python dtype: `Dict[str, Union[torch.Tensor, int]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LoadAudio:
    SUPPORTED_FORMATS = ('.wav', '.mp3', '.ogg', '.flac', '.aiff', '.aif')

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [
            f for f in os.listdir(input_dir)
            if (os.path.isfile(os.path.join(input_dir, f))
                and f.endswith(LoadAudio.SUPPORTED_FORMATS)
            )
        ]
        return {"required": {"audio": (sorted(files), {"audio_upload": True})}}

    CATEGORY = "_for_testing/audio"

    RETURN_TYPES = ("AUDIO", )
    FUNCTION = "load"

    def load(self, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        waveform, sample_rate = torchaudio.load(audio_path)
        multiplier = 1.0
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return (audio, )

    @classmethod
    def IS_CHANGED(s, audio):
        image_path = folder_paths.get_annotated_filepath(audio)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, audio):
        if not folder_paths.exists_annotated_filepath(audio):
            return "Invalid audio file: {}".format(audio)
        return True

```
