---
tags:
- Prompt
- PromptStyling
---

# SDXL Prompt Styler Advanced
## Documentation
- Class name: `SDXLPromptStylerAdvanced`
- Category: `utils`
- Output node: `False`

The SDXLPromptStylerAdvanced node is designed to process and style text prompts using advanced templates. It allows for the customization of positive and negative prompts with specific styles, and supports logging of the styled prompts for debugging or analysis purposes. This node is capable of handling more complex prompt styling scenarios, including the differentiation between global and local positive prompts, as well as the selective application of negative prompts.
## Input types
### Required
- **`text_positive_g`**
    - The global positive text to be styled. This text is intended for use across multiple prompts or contexts, providing a consistent positive message.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text_positive_l`**
    - The local positive text to be styled. This text is specific to a particular prompt or context, allowing for more targeted positive messaging.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text_negative`**
    - The negative text to be styled. This text is used to provide a contrasting or opposing viewpoint to the positive text, enhancing the prompt's effectiveness.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`style`**
    - The style identifier used to select the template for styling the prompts. This allows for the customization of prompt appearance according to predefined styles.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`negative_prompt_to`**
    - Determines the application scope of the negative prompt, whether it applies globally, locally, or both, influencing how the negative text is integrated into the styled prompts.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`log_prompt`**
    - A flag indicating whether the styled prompts should be logged to the console. This is useful for debugging or analyzing the styling process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`text_positive_g`**
    - Comfy dtype: `STRING`
    - The styled global positive text, reflecting the application of the selected style.
    - Python dtype: `str`
- **`text_positive_l`**
    - Comfy dtype: `STRING`
    - The styled local positive text, tailored to a specific context or prompt.
    - Python dtype: `str`
- **`text_positive`**
    - Comfy dtype: `STRING`
    - The combined styled positive text, incorporating both global and local positive messages.
    - Python dtype: `str`
- **`text_negative_g`**
    - Comfy dtype: `STRING`
    - The styled global negative text, applicable when the negative prompt is set to affect global context.
    - Python dtype: `str`
- **`text_negative_l`**
    - Comfy dtype: `STRING`
    - The styled local negative text, applicable when the negative prompt is set to affect local context.
    - Python dtype: `str`
- **`text_negative`**
    - Comfy dtype: `STRING`
    - The combined styled negative text, integrating both global and local negative messages.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SDXLPromptStylerAdvanced:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        current_directory = os.path.dirname(os.path.realpath(__file__)) + "/data"
        self.json_data, styles = load_styles_from_directory(current_directory)

        return {
            "required": {
                "text_positive_g": ("STRING", {"default": "", "multiline": True}),
                "text_positive_l": ("STRING", {"default": "", "multiline": True}),
                "text_negative": ("STRING", {"default": "", "multiline": True}),
                "style": (styles,),
                "negative_prompt_to": (["Both", "G only", "L only"], {"default": "Both"}),
                "log_prompt": (["No", "Yes"], {"default": "No"}),
            },
        }

    RETURN_TYPES = ('STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING',)
    RETURN_NAMES = (
        'text_positive_g', 'text_positive_l', 'text_positive', 'text_negative_g', 'text_negative_l', 'text_negative',)
    FUNCTION = 'prompt_styler_advanced'
    CATEGORY = 'utils'

    def prompt_styler_advanced(self, text_positive_g, text_positive_l, text_negative, style, negative_prompt_to,
                               log_prompt):
        # Process and combine prompts in templates
        # The function replaces the positive prompt placeholder in the template,
        # and combines the negative prompt with the template's negative prompt, if they exist.
        text_positive_g_styled, text_positive_l_styled, text_positive_styled, text_negative_g_styled, text_negative_l_styled, text_negative_styled = read_sdxl_templates_replace_and_combine_advanced(
            self.json_data, style, text_positive_g, text_positive_l, text_negative, negative_prompt_to)

        # If logging is enabled (log_prompt is set to "Yes"), 
        # print the style, positive and negative text, and positive and negative prompts to the console
        if log_prompt == "Yes":
            print(f"style: {style}")
            print(f"text_positive_g: {text_positive_g}")
            print(f"text_positive_l: {text_positive_l}")
            print(f"text_negative: {text_negative}")
            print(f"text_positive_g_styled: {text_positive_g_styled}")
            print(f"text_positive_l_styled: {text_positive_l_styled}")
            print(f"text_positive_styled: {text_positive_styled}")
            print(f"text_negative_g_styled: {text_negative_g_styled}")
            print(f"text_negative_l_styled: {text_negative_l_styled}")
            print(f"text_negative_styled: {text_negative_styled}")

        return text_positive_g_styled, text_positive_l_styled, text_positive_styled, text_negative_g_styled, text_negative_l_styled, text_negative_styled

```
