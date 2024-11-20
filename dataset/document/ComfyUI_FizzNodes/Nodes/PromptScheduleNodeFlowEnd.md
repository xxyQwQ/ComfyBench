---
tags:
- AnimationScheduling
- Scheduling
---

# Prompt Schedule NodeFlow End 📅🅕🅝
## Documentation
- Class name: `PromptScheduleNodeFlowEnd`
- Category: `FizzNodes 📅🅕🅝/ScheduleNodes`
- Output node: `False`

This node represents the final step in a scheduling node flow, designed to evaluate and finalize the JSON structure produced by preceding nodes. It focuses on ensuring the integrity of the JSON output, making adjustments as necessary to fit the expected format for downstream processing.
## Input types
### Required
- **`text`**
    - The JSON text to be evaluated and potentially modified for format correctness. It plays a crucial role in the node's operation by serving as the primary data that will be adjusted to meet the expected output structure.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - A clip parameter that influences the animation and scheduling process, affecting the final output based on the clip's characteristics.
    - Comfy dtype: `CLIP`
    - Python dtype: `Clip`
- **`max_frames`**
    - Specifies the maximum number of frames to be considered in the scheduling process. It affects how the JSON text is evaluated, particularly in terms of frame-related adjustments.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`print_output`**
    - A boolean flag indicating whether the output should be printed, affecting the node's execution by potentially adding a debugging or logging step.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`current_frame`**
    - Indicates the current frame being processed. This parameter is crucial for determining the specific adjustments needed for the JSON text based on the frame's context.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`pre_text`**
    - Optional pre-text that can be prepended to the input text, affecting the context or framing of the animation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text`**
    - Optional appended text that can be added to the input text, modifying the final animation's context or narrative.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`pw_a`**
    - A weight parameter influencing the animation's dynamics or characteristics in a specific manner.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_b`**
    - A weight parameter influencing the animation's dynamics or characteristics in a specific manner.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_c`**
    - A weight parameter influencing the animation's dynamics or characteristics in a specific manner.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_d`**
    - A weight parameter influencing the animation's dynamics or characteristics in a specific manner.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`POS`**
    - Comfy dtype: `CONDITIONING`
    - Represents the positive conditioning output, which is part of the final evaluated and adjusted JSON structure, contributing to the animation's positive aspects.
    - Python dtype: `Conditioning`
- **`NEG`**
    - Comfy dtype: `CONDITIONING`
    - Represents the negative conditioning output, which is part of the final evaluated and adjusted JSON structure, contributing to the animation's negative aspects.
    - Python dtype: `Conditioning`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class PromptScheduleNodeFlowEnd:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": False, "forceInput": True}), 
                            "clip": ("CLIP", ),
                            "max_frames": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0,}),
                            "print_output": ("BOOLEAN", {"default": False}),
                            "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0, "forceInput": True}),},
               "optional": {"pre_text": ("STRING", {"multiline": True, "forceInput": True}),
                            "app_text": ("STRING", {"multiline": True, "forceInput": True}),
                            "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                            "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                            "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                            "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                            }}
    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("POS", "NEG",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"

    def animate(self, text, max_frames, print_output, current_frame, clip, pw_a = 0, pw_b = 0, pw_c = 0, pw_d = 0, pre_text = '', app_text = ''):
        if text[-1] == ",":
            text = text[:-1]
        if text[0] == ",":
            text = text[:0]
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
        return prompt_schedule(settings, clip)

```
