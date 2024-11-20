---
tags:
- AnimationScheduling
- Scheduling
---

# Batch String Schedule 📅🅕🅝
## Documentation
- Class name: `BatchStringSchedule`
- Category: `FizzNodes 📅🅕🅝/BatchScheduleNodes`
- Output node: `False`

The `BatchStringSchedule` node processes animation prompts by splitting them into positive and negative prompts, interpolating these prompts over a series of frames, and returning them as a batch of strings. This node is designed to handle the temporal dynamics of animation prompts, facilitating the generation of text sequences that vary over time.
## Input types
### Required
- **`text`**
    - The text input represents the base animation prompt that will be split into positive and negative prompts, serving as the foundation for the temporal interpolation process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`max_frames`**
    - Specifies the maximum number of frames for which the prompts will be interpolated, defining the temporal extent of the animation sequence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`print_output`**
    - A flag indicating whether the output should be printed, affecting the node's output behavior.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`pre_text`**
    - Text to be prepended to each animation prompt, modifying the initial content of the prompts before they are split and interpolated.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text`**
    - Text to be appended to each animation prompt, modifying the final content of the prompts before they are split and interpolated.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`pw_a`**
    - Weight parameter A for interpolation, influencing how the positive and negative prompts are blended over time.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_b`**
    - Weight parameter B for interpolation, further influencing the blending of prompts over the animation sequence.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_c`**
    - Weight parameter C for interpolation, adjusting the influence of certain aspects of the prompt interpolation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_d`**
    - Weight parameter D for interpolation, adjusting the influence of certain aspects of the prompt interpolation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`POS`**
    - Comfy dtype: `STRING`
    - The interpolated positive prompt for the current frame, as part of the batch of strings.
    - Python dtype: `str`
- **`NEG`**
    - Comfy dtype: `STRING`
    - The interpolated negative prompt for the current frame, as part of the batch of strings.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class BatchStringSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                    "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                    "print_output": ("BOOLEAN", {"default": False}),
            },
                "optional": {
                    "pre_text": ("STRING", {"multiline": True, "forceInput": True}),
                    "app_text": ("STRING", {"multiline": True, "forceInput": True}),
                    "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                    "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                    "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                    "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("POS", "NEG",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, text, max_frames,  pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text='',
                print_output=False):
        settings = ScheduleSettings(
            text_g=text,
            pre_text_G=pre_text,
            app_text_G=app_text,
            text_L=None,
            pre_text_L=None,
            app_text_L=None,
            max_frames=max_frames,
            current_frame=None,
            print_output=print_output,
            pw_a=pw_a,
            pw_b=pw_b,
            pw_c=pw_c,
            pw_d=pw_d,
            start_frame=0,
            end_frame=0,
            width=None,
            height=None,
            crop_w=None,
            crop_h=None,
            target_width=None,
            target_height=None,
        )
        return batch_string_schedule(settings)

```
