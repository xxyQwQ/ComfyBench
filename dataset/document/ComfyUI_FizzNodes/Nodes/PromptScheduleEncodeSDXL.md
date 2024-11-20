---
tags:
- AnimationScheduling
- PromptScheduling
- Scheduling
---

# Prompt Schedule SDXL 📅🅕🅝
## Documentation
- Class name: `PromptScheduleEncodeSDXL`
- Category: `FizzNodes 📅🅕🅝/ScheduleNodes`
- Output node: `False`

This node is designed to schedule and encode prompts specifically for the SDXL model, handling the process of tokenization, applying weighted combinations of prompts, and returning the conditioned output for either the current, next, or an averaged state. It focuses on enhancing the flexibility and precision of prompt conditioning in generative models.
## Input types
### Required
- **`width`**
    - Specifies the width of the image for which the prompt is being scheduled, affecting the spatial dimensions of the generated output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Specifies the height of the image for which the prompt is being scheduled, affecting the spatial dimensions of the generated output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop_w`**
    - Defines the width of the crop area, which is used to refine the focus of the generated output within the specified width.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop_h`**
    - Defines the height of the crop area, which is used to refine the focus of the generated output within the specified height.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`target_width`**
    - The target width after processing, which may involve resizing or cropping to fit specific output requirements.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`target_height`**
    - The target height after processing, which may involve resizing or cropping to fit specific output requirements.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`text_g`**
    - Represents the global textual content to be scheduled and encoded, serving as a foundational element for generating prompts within the SDXL model.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - Specifies the clip model to be used in conjunction with the prompt for generating or refining outputs, integrating visual context or constraints into the prompt scheduling process.
    - Comfy dtype: `CLIP`
    - Python dtype: `str`
- **`text_l`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
- **`max_frames`**
    - The maximum number of frames to be considered for scheduling, defining the temporal boundary of the animation or video output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`current_frame`**
    - The current frame number in the sequence, used to determine the specific state of prompt conditioning at any given time.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`print_output`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
### Optional
- **`pre_text_G`**
    - Pre-text for global prompts, used to prepend additional context or instructions to the global text, enhancing the specificity of the generated output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text_G`**
    - Append text for global prompts, used to add concluding remarks or instructions to the global text, further refining the output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`pre_text_L`**
    - Pre-text for local prompts, similar to global pre-text but applied to local or specific areas of interest within the generated output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text_L`**
    - Append text for local prompts, similar to global append text but focused on enhancing local or specific areas within the output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`pw_a`**
    - Weight parameter A, part of a set of parameters used to adjust the influence of different prompts or aspects of the prompt on the final output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_b`**
    - Weight parameter B, works in conjunction with other weight parameters to fine-tune the balance between various elements of the prompt.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_c`**
    - Weight parameter C, another factor in the complex equation of prompt weighting, contributing to the nuanced control over the generated output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_d`**
    - Weight parameter D, completes the set of weight parameters, ensuring a comprehensive approach to prompt conditioning.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`POS`**
    - Comfy dtype: `CONDITIONING`
    - The enhanced positive prompt, conditioned and ready for use in generating desired outputs with the SDXL model.
    - Python dtype: `str`
- **`NEG`**
    - Comfy dtype: `CONDITIONING`
    - The adjusted negative prompt, conditioned to minimize its influence on the generated outputs.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class PromptScheduleEncodeSDXL:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "target_width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "target_height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "text_g": ("STRING", {"multiline": True, }), "clip": ("CLIP", ),
                "text_l": ("STRING", {"multiline": True, }), "clip": ("CLIP", ),
                "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0}),
                "print_output":("BOOLEAN", {"default": False}),
        },
            "optional": {
                "pre_text_G": ("STRING", {"multiline": True, "forceInput": True}),
                "app_text_G": ("STRING", {"multiline": True, "forceInput": True}),
                "pre_text_L": ("STRING", {"multiline": True, "forceInput": True}),
                "app_text_L": ("STRING", {"multiline": True, "forceInput": True}),
                "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
            }
        }
    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("POS", "NEG",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"

    def animate(self, clip, text_g, text_l, width, height, crop_w, crop_h, target_width, target_height, max_frames, current_frame, print_output, app_text_G = '', app_text_L = '', pre_text_G = '', pre_text_L = '', pw_a=0, pw_b=0, pw_c=0, pw_d=0):
        settings = ScheduleSettings(
            text_g=text_g,
            pre_text_G=pre_text_G,
            app_text_G=app_text_G,
            text_L=text_l,
            pre_text_L=pre_text_L,
            app_text_L=app_text_L,
            max_frames=max_frames,
            current_frame=current_frame,
            print_output=print_output,
            pw_a=pw_a,
            pw_b=pw_b,
            pw_c=pw_c,
            pw_d=pw_d,
            start_frame=0,
            end_frame=0,
            width=width,
            height=height,
            crop_w=crop_w,
            crop_h=crop_h,
            target_width=target_width,
            target_height=target_height,
        )
        return prompt_schedule_SDXL(settings,clip)

```
