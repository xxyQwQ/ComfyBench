---
tags:
- AnimationScheduling
- PromptScheduling
- Scheduling
---

# Batch Prompt Schedule 📅🅕🅝
## Documentation
- Class name: `BatchPromptSchedule`
- Category: `FizzNodes 📅🅕🅝/BatchScheduleNodes`
- Output node: `False`

The BatchPromptSchedule node is designed to process animation prompts by cleaning up input text, splitting it into positive and negative prompts based on predefined weights, and then applying a series of transformations to interpolate and condition these prompts for animation sequences. This node facilitates the generation of dynamic, frame-specific animation content by leveraging prompt scheduling techniques to adjust the influence of various prompts over the course of an animation.
## Input types
### Required
- **`text`**
    - The text parameter includes the input text for animation prompts, which is processed to remove whitespace and newlines before being split into positive and negative prompts.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - The clip parameter represents the animation clip to which the prompt scheduling and conditioning will be applied. It is essential for aligning the generated prompts with the specific frames and dynamics of the animation.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.Tensor`
- **`max_frames`**
    - This parameter specifies the maximum number of frames for the animation, used to determine the scheduling and interpolation of prompts across the animation sequence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`print_output`**
    - A boolean parameter indicating whether to print the output of the prompt processing for debugging or informational purposes.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`pre_text`**
    - Pre-text to be added before each prompt, used to modify or enhance the original prompt text before processing.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text`**
    - Appended text to be added after each prompt, serving to further customize or refine the prompt text.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`start_frame`**
    - Specifies the starting frame number for the animation, used in conjunction with end_frame to define the range of frames affected by the prompt scheduling.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_frame`**
    - Defines the ending frame number for the animation, setting the boundary for prompt processing and application.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`pw_a`**
    - Parameter weight A, part of a set of weights used to adjust the influence of various prompts or conditions within the animation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_b`**
    - Parameter weight B, another weight in the set for fine-tuning the prompt conditioning and scheduling.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_c`**
    - Parameter weight C, used alongside other weights to customize the prompt's impact on the animation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_d`**
    - Parameter weight D, the last of the set of weights for adjusting prompt influence, contributing to the dynamic generation of animation content.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`POS`**
    - Comfy dtype: `CONDITIONING`
    - The output 'POS' represents the conditioned positive prompts, ready for use in generating or influencing animation frames.
    - Python dtype: `torch.Tensor`
- **`NEG`**
    - Comfy dtype: `CONDITIONING`
    - The output 'NEG' represents the conditioned negative prompts, similarly prepared for influencing animation frames in contrast to the positive prompts.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [ControlNetApplyAdvanced](../../Comfy/Nodes/ControlNetApplyAdvanced.md)
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [Prompts Everywhere](../../cg-use-everywhere/Nodes/Prompts Everywhere.md)
    - [ACN_AdvancedControlNetApply](../../ComfyUI-Advanced-ControlNet/Nodes/ACN_AdvancedControlNetApply.md)
    - LinearBatchCreativeInterpolation



## Source code
```python
class BatchPromptSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                    "clip": ("CLIP",),
                    "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                    "print_output":("BOOLEAN", {"default": False}),
                },
                "optional": {
                    "pre_text": ("STRING", {"multiline": True, "forceInput": True}),
                    "app_text": ("STRING", {"multiline": True, "forceInput": True}),
                    "start_frame": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1, "display": "start_frame(print_only)", }),
                    "end_frame": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1, "display": "end_frame(print_only)",}),
                    "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                    "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                    "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                    "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("POS", "NEG",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, text, max_frames, print_output, clip, start_frame, end_frame, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text=''
    ):
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
            start_frame=start_frame,
            end_frame=end_frame,
            width=None,
            height=None,
            crop_w=None,
            crop_h=None,
            target_width=None,
            target_height=None,
        )
        return batch_prompt_schedule(settings, clip)

```
