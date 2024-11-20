---
tags:
- AnimationScheduling
- PromptScheduling
- Scheduling
---

# Batch Prompt Schedule (Latent Input) 📅🅕🅝
## Documentation
- Class name: `BatchPromptScheduleLatentInput`
- Category: `FizzNodes 📅🅕🅝/BatchScheduleNodes`
- Output node: `False`

This node is designed to process animation prompts for generating scheduled latent inputs. It sequences the user's formatted prompt into current and next prompts along with their conditioning strengths, evaluates expressions within the prompts, and applies a scheduling algorithm to produce a batch of conditionings tailored to the input latent. The node's functionality facilitates the creation of dynamic, time-sequenced visual content by manipulating latent space representations in accordance with the specified prompt schedule.
## Input types
### Required
- **`text`**
    - The user's input prompt intended for animation, which is processed to generate positive and negative prompts for content generation. This input is crucial for defining the thematic and narrative direction of the generated content.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - A representation of the clip model used for conditioning the input prompt, playing a key role in the generation process by influencing the visual characteristics of the output.
    - Comfy dtype: `CLIP`
    - Python dtype: `str`
- **`num_latents`**
    - Specifies the number of latent vectors to be used in the generation process, directly impacting the diversity and variation of the generated content.
    - Comfy dtype: `LATENT`
    - Python dtype: `int`
- **`print_output`**
    - A boolean flag that, when set, enables the printing of output for debugging purposes, aiding in the analysis and refinement of the generation process.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`pre_text`**
    - Pre-text added to the animation prompts before processing, influencing the initial context and direction of the generated content.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`app_text`**
    - Appended text to the animation prompts, modifying the final output by adding specific details or themes to the generated content.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`start_frame`**
    - Defines the starting frame for the animation, setting the initial point for the prompt scheduling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_frame`**
    - Specifies the ending frame for the animation, marking the conclusion of the prompt scheduling.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`pw_a`**
    - A parameter weight influencing the conditioning strength of the animation, affecting the intensity of the generated content's attributes.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_b`**
    - A parameter weight that adjusts the balance between different attributes in the generated content, impacting the overall visual outcome.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_c`**
    - A parameter weight used to fine-tune the conditioning process, allowing for precise control over the generated content's characteristics.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pw_d`**
    - A parameter weight that affects the temporal dynamics of the animation, influencing how attributes evolve over time.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`POS`**
    - Comfy dtype: `CONDITIONING`
    - The processed and scheduled positive conditioning, ready for use in generating content that emphasizes desired attributes.
    - Python dtype: `str`
- **`NEG`**
    - Comfy dtype: `CONDITIONING`
    - The processed and scheduled negative conditioning, aimed at minimizing or avoiding certain attributes in the generated content.
    - Python dtype: `str`
- **`INPUT_LATENTS`**
    - Comfy dtype: `LATENT`
    - The original input latents passed through the node, potentially modified or unchanged, depending on the node's internal logic.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [ConditioningCombine](../../Comfy/Nodes/ConditioningCombine.md)
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [KSampler Adv. (Efficient)](../../efficiency-nodes-comfyui/Nodes/KSampler Adv. (Efficient).md)



## Source code
```python
class BatchPromptScheduleLatentInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                    "clip": ("CLIP",),
                    "num_latents": ("LATENT", ),
                    "print_output":("BOOLEAN", {"default": False}),
                },
                "optional": {"pre_text": ("STRING", {"multiline": True, "forceInput": True}),
                    "app_text": ("STRING", {"multiline": True, "forceInput": True}),
                    "start_frame": ("INT", {"default": 0.0, "min": 0, "max": 9999, "step": 1, "display": "start_frame(print_only)", }),
                    "end_frame": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1, "display": "end_frame(print_only)", }),
                    "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                    "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                    "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                    "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT",) #"CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("POS", "NEG", "INPUT_LATENTS", ) #"POS_CUR", "NEG_CUR", "POS_NXT", "NEG_NXT",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, text, num_latents, print_output, clip, start_frame, end_frame, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text=''
    ):
        settings = ScheduleSettings(
            text_g=text,
            pre_text_G=pre_text,
            app_text_G=app_text,
            text_L=None,
            pre_text_L=None,
            app_text_L=None,
            max_frames=sum(tensor.size(0) for tensor in num_latents.values()),
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
        return batch_prompt_schedule_latentInput(settings,clip, num_latents)

```
