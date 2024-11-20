---
tags:
- AnimationScheduling
- PromptScheduling
- Scheduling
---

# Batch Prompt Schedule SDXL (Latent Input) 📅🅕🅝
## Documentation
- Class name: `BatchPromptScheduleSDXLLatentInput`
- Category: `FizzNodes 📅🅕🅝/BatchScheduleNodes`
- Output node: `False`

This node is designed to process animation prompts for both G and L clips, applying a scheduling algorithm that separates, tokenizes, and then recombines these prompts with weighted adjustments before finally applying a batch of conditionings. It uniquely handles latent inputs, allowing for the dynamic adjustment of animation parameters based on the provided latents.
## Input types
### Required
- **`width`**
    - The width of the clip or frame in pixels.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The height of the clip or frame in pixels.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop_w`**
    - The width of the crop area in pixels.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop_h`**
    - The height of the crop area in pixels.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`target_width`**
    - The target width of the clip or frame after processing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`target_height`**
    - The target height of the clip or frame after processing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`text_g`**
    - The text prompt for G clips, which will be processed, tokenized, and adjusted according to the scheduling algorithm.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - Represents the clip information or parameters that are used in conjunction with the text prompts to apply the scheduling algorithm.
    - Comfy dtype: `CLIP`
    - Python dtype: `ClipType`
- **`text_l`**
    - The text prompt for L clips, similar to text_g, it will be processed, tokenized, and adjusted for scheduling.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`num_latents`**
    - The number of latent inputs that are used to dynamically adjust the animation parameters during the scheduling process.
    - Comfy dtype: `LATENT`
    - Python dtype: `int`
- **`print_output`**
    - A flag indicating whether to print the output of the scheduling process.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`pre_text_G`**
    - Pre-text to be added to the G clip's text prompt before processing.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text_G`**
    - App-text to be appended to the G clip's text prompt after processing.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`pre_text_L`**
    - Pre-text to be added to the L clip's text prompt before processing.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text_L`**
    - App-text to be appended to the L clip's text prompt after processing.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`pw_a`**
    - Weight parameter a, used in the scheduling algorithm for adjusting prompts.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_b`**
    - Weight parameter b, used in the scheduling algorithm for adjusting prompts.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_c`**
    - Weight parameter c, used in the scheduling algorithm for adjusting prompts.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_d`**
    - Weight parameter d, used in the scheduling algorithm for adjusting prompts.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`POS`**
    - Comfy dtype: `CONDITIONING`
    - The positive conditioning batch resulting from the applied scheduling and animation adjustments.
    - Python dtype: `ConditioningBatch`
- **`NEG`**
    - Comfy dtype: `CONDITIONING`
    - The negative conditioning batch resulting from the applied scheduling and animation adjustments.
    - Python dtype: `ConditioningBatch`
- **`POS_CUR`**
    - Comfy dtype: `LATENT`
    - The current positive prompt conditioning, part of the output detailing the specific conditioning state at the current animation frame.
    - Python dtype: `ConditioningBatch`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class BatchPromptScheduleEncodeSDXLLatentInput:
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
                "num_latents": ("LATENT", ),
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
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT",)# "CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("POS", "NEG", "POS_CUR", "NEG_CUR", "POS_NXT", "NEG_NXT",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, clip, text_g, text_l, width, height, crop_w, crop_h, target_width, target_height, num_latents, print_output, app_text_G = '', app_text_L = '', pre_text_G = '', pre_text_L = '', pw_a=0, pw_b=0, pw_c=0, pw_d=0):
        settings = ScheduleSettings(
            text_g=text_g,
            pre_text_G=pre_text_G,
            app_text_G=app_text_G,
            text_L=text_l,
            pre_text_L=pre_text_L,
            app_text_L=app_text_L,
            max_frames=sum(tensor.size(0) for tensor in num_latents.values()),
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
        return batch_prompt_schedule_SDXL_latentInput(settings, clip, num_latents)

```
