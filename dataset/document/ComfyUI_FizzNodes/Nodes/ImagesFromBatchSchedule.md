---
tags:
- AnimationScheduling
- Scheduling
---

# Image Select Schedule 📅🅕🅝
## Documentation
- Class name: `ImagesFromBatchSchedule`
- Category: `FizzNodes 📅🅕🅝/ScheduleNodes`
- Output node: `False`

This node is designed to generate a batch of images based on a predefined schedule. It leverages a batch value schedule list to select and export images from an input batch, facilitating the creation of image sequences or animations. The node's functionality is rooted in the ability to process and manipulate image batches according to specified scheduling criteria, making it a valuable tool for dynamic image generation and manipulation tasks.
## Input types
### Required
- **`images`**
    - Specifies the input batch of images to be processed and scheduled for output. This parameter is crucial for defining the source images that will be manipulated and organized according to the batch schedule.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`text`**
    - Defines the scheduling criteria or instructions as a text input, used to determine how images are selected and organized in the output batch.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`current_frame`**
    - Indicates the current frame number in the sequence, used to track progression and manage image selection according to the schedule.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`max_frames`**
    - Sets the maximum number of frames to be included in the output batch, limiting the sequence length according to this threshold.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`print_output`**
    - A boolean flag that, when set to True, enables the printing of output information for debugging or informational purposes.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a batch of images that have been selected and organized according to the specified schedule, ready for further processing or visualization.
    - Python dtype: `List[torch.Tensor]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ImagesFromBatchSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "text": ("STRING", {"multiline": True, "default":defaultPrompt}),
                "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0, }),
                "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                "print_output": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "animate"
    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"

    def animate(self, images, text, current_frame, max_frames, print_output):
        inputText = str("{" + text + "}")
        inputText = re.sub(r',\s*}', '}', inputText)
        animation_prompts = json.loads(inputText.strip())
        pos_cur_prompt, pos_nxt_prompt, weight = interpolate_prompt_series(animation_prompts, max_frames, 0, "",
                                                                           "", 0,
                                                                           0, 0, 0, print_output)
        selImages = selectImages(images,pos_cur_prompt[current_frame])
        return selImages

```
