---
tags:
- Checkpoint
---

# Checkpoint Name Selector
## Documentation
- Class name: `CheckpointNameSelector`
- Category: `Art Venture/Utils`
- Output node: `False`

The CheckpointNameSelector node is designed to facilitate the selection of checkpoint names from a predefined list, streamlining the process of identifying and utilizing specific checkpoints within a workflow. It abstracts the complexity of navigating through checkpoint directories, offering a user-friendly interface for selecting the desired checkpoint name.
## Input types
### Required
- **`ckpt_name`**
    - The 'ckpt_name' parameter allows users to specify the name of the checkpoint they wish to select. This selection is crucial as it determines which checkpoint will be utilized for subsequent operations, directly influencing the workflow's outcome.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
## Output types
- **`ckpt_name`**
    - Comfy dtype: `COMBO[STRING]`
    - Returns the selected checkpoint name, enabling further processing or utilization within the workflow.
    - Python dtype: `str`
- **`ckpt_name_str`**
    - Comfy dtype: `STRING`
    - Provides the selected checkpoint name as a string, facilitating its use in scenarios where a string representation is required.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class UtilCheckpointSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = (folder_paths.get_filename_list("checkpoints"), "STRING")
    RETURN_NAMES = ("ckpt_name", "ckpt_name_str")
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "get_ckpt_name"

    def get_ckpt_name(self, ckpt_name):
        return (ckpt_name, ckpt_name)

```
