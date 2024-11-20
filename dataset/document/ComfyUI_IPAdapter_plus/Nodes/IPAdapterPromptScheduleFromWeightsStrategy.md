---
tags:
- ControlNet
- Weight
---

# Prompt Schedule From Weights Strategy
## Documentation
- Class name: `IPAdapterPromptScheduleFromWeightsStrategy`
- Category: `ipadapter/weights`
- Output node: `False`

This node generates a prompt schedule based on a given weights strategy, dynamically adjusting the prompt sequence to align with the specified frame count and incorporating optional starting and ending frames. It's designed to tailor the prompt delivery in image processing tasks, ensuring that prompts are optimally distributed across the desired number of frames.
## Input types
### Required
- **`weights_strategy`**
    - A strategy dictating the distribution of weights across frames, including the total number of frames, and whether to add starting or ending frames. This strategy influences how the prompt schedule is generated, ensuring alignment with the frame-based processing requirements.
    - Comfy dtype: `WEIGHTS_STRATEGY`
    - Python dtype: `Dict[str, Union[int, bool]]`
- **`prompt`**
    - A multiline string input that represents the base prompt to be scheduled. This input is split and adjusted according to the weights strategy to fit the frame count, enhancing the relevance of the prompt at each frame.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`prompt_schedule`**
    - Comfy dtype: `STRING`
    - A string representation of the scheduled prompts, formatted to align with the frame distribution specified by the weights strategy. This schedule is ready for use in frame-based image processing tasks.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class IPAdapterPromptScheduleFromWeightsStrategy():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "weights_strategy": ("WEIGHTS_STRATEGY",),
            "prompt": ("STRING", {"default": "", "multiline": True }),
            }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_schedule", )
    FUNCTION = "prompt_schedule"
    CATEGORY = "ipadapter/weights"

    def prompt_schedule(self, weights_strategy, prompt=""):
        frames = weights_strategy["frames"]
        add_starting_frames = weights_strategy["add_starting_frames"]
        add_ending_frames = weights_strategy["add_ending_frames"]
        frame_count = weights_strategy["frame_count"]

        out = ""

        prompt = [p for p in prompt.split("\n") if p.strip() != ""]

        if len(prompt) > 0 and frame_count > 0:
            # prompt_pos must be the same size as the image batch
            if len(prompt) > frame_count:
                prompt = prompt[:frame_count]
            elif len(prompt) < frame_count:
                prompt += [prompt[-1]] * (frame_count - len(prompt))

            if add_starting_frames > 0:
                out += f"\"0\": \"{prompt[0]}\",\n"
            for i in range(frame_count):
                out += f"\"{i * frames + add_starting_frames}\": \"{prompt[i]}\",\n"
            if add_ending_frames > 0:
                out += f"\"{frame_count * frames + add_starting_frames}\": \"{prompt[-1]}\",\n"

        return (out, )

```
