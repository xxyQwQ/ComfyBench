---
tags:
- AnimationScheduling
- PromptScheduling
- Scheduling
---

# Batch Prompt Schedule SDXL 📅🅕🅝
## Documentation
- Class name: `BatchPromptScheduleEncodeSDXL`
- Category: `FizzNodes 📅🅕🅝/BatchScheduleNodes`
- Output node: `False`

This node is designed to process and encode batch prompts for Stable Diffusion XL models, allowing for the scheduling of G and L clips separately before tokenization. It integrates a weighted addition process to generate a batch of conditionings tailored for animation or dynamic content creation.
## Input types
### Required
- **`width`**
    - Specifies the width of the output image or frame, impacting the aspect ratio and detail level of the generated content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Determines the height of the output image or frame, affecting the aspect ratio and overall composition of the generated content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop_w`**
    - Defines the width of the cropping area, used to focus on or exclude specific parts of the generated content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop_h`**
    - Specifies the height of the cropping area, enabling precise control over the content's framing and focus areas.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`target_width`**
    - The desired width for resizing the output, allowing for adjustments to the content's dimensions without altering its aspect ratio.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`target_height`**
    - The target height for resizing the output, facilitating dimension adjustments while maintaining the original aspect ratio.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`text_g`**
    - Represents the global text prompts that guide the overall theme and content generation, serving as a foundational element for the animation or dynamic content.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - Refers to the clip model used for conditioning the prompts, playing a critical role in the interpretation and processing of the text inputs.
    - Comfy dtype: `CLIP`
    - Python dtype: `str`
- **`text_l`**
    - Denotes the local text prompts that provide detailed guidance for specific parts of the content, complementing the global prompts to refine the output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`max_frames`**
    - Indicates the maximum number of frames to be generated, defining the length and scope of the animation or dynamic content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`print_output`**
    - A flag to enable or disable the printing of output for debugging or transparency purposes during the content generation process.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`pre_text_G`**
    - Prepended text for global prompts, used to add context or modify the tone of the global text inputs before processing.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text_G`**
    - Appended text for global prompts, allowing for additional details or thematic elements to be included after the main global text.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`pre_text_L`**
    - Prepended text for local prompts, used to introduce or alter the context of the local text inputs, enhancing specificity and focus.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text_L`**
    - Appended text for local prompts, providing a means to extend or refine the local text inputs with further details or thematic elements.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`pw_a`**
    - Weight parameter A, part of a set of weights used to adjust the influence of different components in the prompt processing and conditioning.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_b`**
    - Weight parameter B, contributing to the fine-tuning of prompt influence and conditioning in the generation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_c`**
    - Weight parameter C, involved in balancing the effects of various prompt components on the content output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_d`**
    - Weight parameter D, used to adjust the relative impact of prompt elements, aiding in the customization of the generation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`POS`**
    - Comfy dtype: `CONDITIONING`
    - The enhanced positive conditioning output, reflecting the emphasized aspects of the original prompt after processing.
    - Python dtype: `List[str]`
- **`NEG`**
    - Comfy dtype: `CONDITIONING`
    - The adjusted negative conditioning output, indicating the de-emphasized elements of the original prompt following processing.
    - Python dtype: `List[str]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class BatchPromptScheduleEncodeSDXL:
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

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)# "CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("POS", "NEG", "POS_CUR", "NEG_CUR", "POS_NXT", "NEG_NXT",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, clip, text_g, text_l, width, height, crop_w, crop_h, target_width, target_height, max_frames, print_output, app_text_G = '', app_text_L = '', pre_text_G = '', pre_text_L = '', pw_a=0, pw_b=0, pw_c=0, pw_d=0):
        settings = ScheduleSettings(
            text_g=text_g,
            pre_text_G=pre_text_G,
            app_text_G=app_text_G,
            text_L=text_l,
            pre_text_L=pre_text_L,
            app_text_L=app_text_L,
            max_frames=max_frames,
            current_frame=None,
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
        return batch_prompt_schedule_SDXL(settings, clip)

```
