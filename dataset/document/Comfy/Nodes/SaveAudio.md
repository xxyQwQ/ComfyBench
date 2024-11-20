# SaveAudio
## Documentation
- Class name: `SaveAudio`
- Category: `_for_testing/audio`
- Output node: `True`

The SaveAudio node is designed for saving audio data to disk in FLAC format. It allows for the inclusion of metadata, such as prompts and additional information, within the saved audio files. This node is particularly useful for persisting audio outputs with contextual metadata, facilitating easier organization and retrieval of generated audio content.
## Input types
### Required
- **`audio`**
    - The audio data to be saved. This includes the waveform and sample rate of the audio to be persisted.
    - Comfy dtype: `AUDIO`
    - Python dtype: `Dict[str, Union[torch.Tensor, int]]`
- **`filename_prefix`**
    - An optional prefix for the filename under which the audio will be saved, allowing for custom naming conventions.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`ui`**
    - Provides a user interface element to display the saved audio files, including filenames and locations.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaveAudio:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO", ),
                              "filename_prefix": ("STRING", {"default": "audio/ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"

    OUTPUT_NODE = True

    CATEGORY = "_for_testing/audio"

    def save_audio(self, audio, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        results = list()

        metadata = {}
        if not args.disable_metadata:
            if prompt is not None:
                metadata["prompt"] = json.dumps(prompt)
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        for (batch_number, waveform) in enumerate(audio["waveform"]):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.flac"

            buff = io.BytesIO()
            torchaudio.save(buff, waveform, audio["sample_rate"], format="FLAC")

            buff = insert_or_replace_vorbis_comment(buff, metadata)

            with open(os.path.join(full_output_folder, file), 'wb') as f:
                f.write(buff.getbuffer())

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "audio": results } }

```
