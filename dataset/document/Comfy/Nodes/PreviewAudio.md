# PreviewAudio
## Documentation
- Class name: `PreviewAudio`
- Category: `_for_testing/audio`
- Output node: `True`

The PreviewAudio node is designed for creating temporary audio previews. It generates audio files with a temporary marker and stores them in a designated temporary directory, facilitating the review or testing of audio content before final processing or saving.
## Input types
### Required
- **`audio`**
    - The primary audio input required for generating a preview. This input is essential for the node to process and create a temporary audio file.
    - Comfy dtype: `AUDIO`
    - Python dtype: `torch.Tensor`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class PreviewAudio(SaveAudio):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"audio": ("AUDIO", ), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

```
