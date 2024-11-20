---
tags:
- Multimedia
- VideoHelperSuite
---

# Meta Batch Manager 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_BatchManager`
- Category: `Video Helper Suite 🎥🅥🅗🅢`
- Output node: `False`

The BatchManager node is designed to manage and orchestrate the processing of video data in batches. It facilitates the efficient handling of large video datasets by breaking them down into smaller, manageable batches for processing, thereby optimizing resource utilization and improving performance.
## Input types
### Required
- **`frames_per_batch`**
    - Specifies the number of frames to be processed in each batch. This parameter is essential for controlling the batch size and thus directly influences the processing efficiency and resource allocation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`meta_batch`**
    - Comfy dtype: `VHS_BatchManager`
    - Returns the updated meta information about the batch after processing, which includes details such as the current state and any data associated with the batch.
    - Python dtype: `CustomType`
- **`ui`**
    - Provides a user interface component indicating the unfinished state of the batch, if applicable.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class BatchManager:
    def __init__(self, frames_per_batch=-1):
        self.frames_per_batch = frames_per_batch
        self.inputs = {}
        self.outputs = {}
        self.unique_id = None
        self.has_closed_inputs = False
        self.total_frames = float('inf')
    def reset(self):
        self.close_inputs()
        for key in self.outputs:
            if getattr(self.outputs[key][-1], "gi_suspended", False):
                try:
                    self.outputs[key][-1].send(None)
                except StopIteration:
                    pass
        self.__init__(self.frames_per_batch)
    def has_open_inputs(self):
        return len(self.inputs) > 0
    def close_inputs(self):
        for key in self.inputs:
            if getattr(self.inputs[key][-1], "gi_suspended", False):
                try:
                    self.inputs[key][-1].send(1)
                except StopIteration:
                    pass
        self.inputs = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "frames_per_batch": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1})
                    },
                "hidden": {
                    "prompt": "PROMPT",
                    "unique_id": "UNIQUE_ID"
                },
                }

    RETURN_TYPES = ("VHS_BatchManager",)
    RETURN_NAMES = ("meta_batch",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "update_batch"

    def update_batch(self, frames_per_batch, prompt=None, unique_id=None):
        if unique_id is not None and prompt is not None:
            requeue = prompt[unique_id]['inputs'].get('requeue', 0)
        else:
            requeue = 0
        if requeue == 0:
            self.reset()
            self.frames_per_batch = frames_per_batch
            self.unique_id = unique_id
        else:
            num_batches = (self.total_frames+self.frames_per_batch-1)//frames_per_batch
            print(f'Meta-Batch {requeue}/{num_batches}')
        #onExecuted seems to not be called unless some message is sent
        return (self,)

```
