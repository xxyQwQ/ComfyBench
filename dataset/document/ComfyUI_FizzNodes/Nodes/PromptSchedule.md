---
tags:
- AnimationScheduling
- PromptScheduling
- Scheduling
---

# Prompt Schedule 📅🅕🅝
## Documentation
- Class name: `PromptSchedule`
- Category: `FizzNodes 📅🅕🅝/ScheduleNodes`
- Output node: `False`

The PromptSchedule node sequences user-formatted prompts, evaluates expressions within these prompts, and applies a scheduling mechanism to generate a batch of conditionings. This process involves sequencing the current and next prompts along with their conditioning strengths, ultimately producing a tailored output that incorporates the scheduled prompt adjustments.
## Input types
### Required
- **`text`**
    - Represents the user's formatted prompt, serving as the primary input for generating conditionings. It is crucial for defining the content and structure of the desired output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - A model or mechanism used for conditioning, influencing the generation process based on the input prompts.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.Tensor`
- **`max_frames`**
    - Specifies the maximum number of frames to be considered for the scheduling process, affecting the temporal scope of the generated outputs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`current_frame`**
    - Indicates the current frame in the sequence, used to determine the specific point of evaluation within the scheduling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`print_output`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
### Optional
- **`pre_text`**
    - Optional pre-text added before the main prompt text, used to prepend additional context or instructions.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text`**
    - Optional appended text added after the main prompt text, used to append additional context or instructions.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`pw_a`**
    - A weight parameter influencing the scheduling process, part of a set of parameters (pw_a, pw_b, pw_c, pw_d) for fine-tuning the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_b`**
    - A weight parameter influencing the scheduling process, part of a set of parameters (pw_a, pw_b, pw_c, pw_d) for fine-tuning the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_c`**
    - A weight parameter influencing the scheduling process, part of a set of parameters (pw_a, pw_b, pw_c, pw_d) for fine-tuning the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_d`**
    - A weight parameter influencing the scheduling process, part of a set of parameters (pw_a, pw_b, pw_c, pw_d) for fine-tuning the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`POS`**
    - Comfy dtype: `CONDITIONING`
    - The positive conditioning output, reflecting the applied prompt schedule and targeted adjustments.
    - Python dtype: `object`
- **`NEG`**
    - Comfy dtype: `CONDITIONING`
    - The negative conditioning output, reflecting the applied prompt schedule and targeted counteractions.
    - Python dtype: `object`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class PromptSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"multiline": True, "default":defaultPrompt}),
                    "clip": ("CLIP", ),
                    "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                    "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0, "forceInput": True }),
                    "print_output":("BOOLEAN", {"default": False,}),
                },
                "optional": {"pre_text": ("STRING", {"multiline": True, "forceInput": True}),
                    "app_text": ("STRING", {"multiline": True, "forceInput": True}),
                    "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                    "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                    "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                    "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("POS", "NEG",)
    FUNCTION = "animate"
    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"

    def animate(self, text, max_frames, print_output, current_frame,clip, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text=''
    ):
        settings = ScheduleSettings(
            text_g=text,
            pre_text_G=pre_text,
            app_text_G=app_text,
            text_L=None,
            pre_text_L=None,
            app_text_L=None,
            max_frames=max_frames,
            current_frame=current_frame,
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
        return prompt_schedule(settings,clip)

```
