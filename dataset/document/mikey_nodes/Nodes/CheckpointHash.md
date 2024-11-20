---
tags:
- Checkpoint
---

# Checkpoint Hash (Mikey)
## Documentation
- Class name: `CheckpointHash`
- Category: `Mikey/Loaders`
- Output node: `False`

The CheckpointHash node is designed to generate a hash value for a given checkpoint file. It identifies the file within a specified directory, computes its hash, and updates a prompt dictionary with the hash and checkpoint name. This functionality is crucial for ensuring the integrity and uniqueness of checkpoint files in machine learning workflows.
## Input types
### Required
- **`ckpt_name`**
    - The name of the checkpoint file for which the hash is to be generated. This parameter is essential for locating the file within the directory and computing its hash.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`ckpt_hash`**
    - Comfy dtype: `STRING`
    - The computed hash value of the checkpoint file, ensuring its integrity and uniqueness.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CheckpointHash:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": ("STRING", {"forceInput": True}),},
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ckpt_hash",)
    FUNCTION = "get_hash"
    CATEGORY = "Mikey/Loaders"

    def get_hash(self, ckpt_name, extra_pnginfo, prompt, unique_id):
        file_list = folder_paths.get_filename_list("checkpoints")
        matching_file = [s for s in file_list if ckpt_name in s][0]
        ckpt_path = folder_paths.get_full_path("checkpoints", matching_file)
        hash = get_file_hash(ckpt_path)
        ckpt_name = os.path.basename(ckpt_name)
        prompt.get(str(unique_id))['inputs']['output_ckpt_hash'] = hash
        prompt.get(str(unique_id))['inputs']['output_ckpt_name'] = ckpt_name
        return (get_file_hash(ckpt_path),)

```
