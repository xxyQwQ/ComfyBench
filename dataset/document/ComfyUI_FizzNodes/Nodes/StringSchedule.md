---
tags:
- AnimationScheduling
- Scheduling
---

# String Schedule 📅🅕🅝
## Documentation
- Class name: `StringSchedule`
- Category: `FizzNodes 📅🅕🅝/ScheduleNodes`
- Output node: `False`

This node is designed for scheduling and animating text and other elements within a given frame range. It supports dynamic text updates, conditional formatting based on frame-specific parameters, and the integration of various inputs to create complex animation sequences. The node facilitates the creation of animated visual content by interpolating between values or states across frames, enabling the development of engaging and dynamic presentations.
## Input types
### Required
- **`text`**
    - The primary text input for animation, serving as the base content or narrative to be animated or modified across frames.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`max_frames`**
    - Specifies the maximum number of frames for the animation, defining the length of the animation sequence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`current_frame`**
    - Indicates the current frame number within the animation sequence, used to determine the specific modifications or animations to apply.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`print_output`**
    - A boolean flag to enable or disable printing of the animation output for debugging or preview purposes.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`pre_text`**
    - Optional text input that precedes the main text, used for adding introductory or contextual content to the animation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text`**
    - Optional text input that appends to the main text, used for adding concluding or additional content to the animation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`pw_a`**
    - A parameter weight used for adjusting or influencing the animation or text modification process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_b`**
    - A parameter weight used for adjusting or influencing the animation or text modification process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_c`**
    - A parameter weight used for adjusting or influencing the animation or text modification process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_d`**
    - A parameter weight used for adjusting or influencing the animation or text modification process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`POS`**
    - Comfy dtype: `STRING`
    - Outputs the positive conditioning or formatted text for the current frame, reflecting dynamic changes or emphasis.
    - Python dtype: `str`
- **`NEG`**
    - Comfy dtype: `STRING`
    - Outputs the negative conditioning or formatted text for the current frame, providing contrast or alternative emphasis.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class StringSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                     "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                     "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0, }),
                     "print_output":("BOOLEAN", {"default": False}),},
                "optional": {"pre_text": ("STRING", {"multiline": True, "forceInput": True}),
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

    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"

    def animate(self, text, max_frames, current_frame, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text='', print_output = False ):
        settings = ScheduleSettings(
        text_g = text,
        pre_text_G = pre_text,
        app_text_G = app_text,
        text_L = None,
        pre_text_L = None,
        app_text_L = None,
        max_frames = max_frames,
        current_frame = current_frame,
        print_output = print_output,
        pw_a = pw_a,
        pw_b = pw_b,
        pw_c = pw_c,
        pw_d = pw_d,
        start_frame = 0,
        end_frame=0,
        width = None,
        height = None,
        crop_w = None,
        crop_h = None,
        target_width = None,
        target_height = None,
        )
        return string_schedule(settings)

```
