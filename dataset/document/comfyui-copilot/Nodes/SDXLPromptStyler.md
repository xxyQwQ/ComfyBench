---
tags:
- Prompt
- PromptStyling
---

# SDXL Prompt Styler
## Documentation
- Class name: `SDXLPromptStyler`
- Category: `utils`
- Output node: `False`

The SDXLPromptStyler node is designed to process and style text prompts based on specified styles and templates. It modifies the input positive and negative prompts by incorporating them into predefined templates, allowing for dynamic and context-aware text generation.
## Input types
### Required
- **`text_positive`**
    - The positive text input that will be styled according to the specified template and style. It plays a crucial role in shaping the output's positive sentiment.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text_negative`**
    - The negative text input that will be combined with the template's negative prompt, if present, to tailor the output's negative sentiment.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`style`**
    - Specifies the styling template to be applied to the text inputs, influencing the thematic presentation of the generated prompts.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`log_prompt`**
    - Controls whether the styling process details are logged, aiding in debugging and process transparency.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`text_positive`**
    - Comfy dtype: `STRING`
    - The styled positive text, transformed according to the specified style and template.
    - Python dtype: `str`
- **`text_negative`**
    - Comfy dtype: `STRING`
    - The styled negative text, adjusted to complement the positive text in accordance with the template's guidelines.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - ShowText|pysssss
    - [CLIPTextEncode](../../Comfy/Nodes/CLIPTextEncode.md)
    - Demofusion From Single File
    - IDGenerationNode
    - [ttN pipeLoader](../../ComfyUI_tinyterraNodes/Nodes/ttN pipeLoader.md)
    - [Eff. Loader SDXL](../../efficiency-nodes-comfyui/Nodes/Eff. Loader SDXL.md)
    - [SDXLPromptStyler](../../comfyui-copilot/Nodes/SDXLPromptStyler.md)
    - ComfyUIStyler
    - NEW_PhotoMaker_Generation
    - Reroute



## Source code
```python
class SDXLPromptStyler:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        current_directory = os.path.dirname(os.path.realpath(__file__)) + "/data"
        self.json_data, styles = load_styles_from_directory(current_directory)

        return {
            "required": {
                "text_positive": ("STRING", {"default": "", "multiline": True}),
                "text_negative": ("STRING", {"default": "", "multiline": True}),
                "style": (styles,),
                "log_prompt": (["No", "Yes"], {"default": "No"}),
            },
        }

    RETURN_TYPES = ('STRING', 'STRING',)
    RETURN_NAMES = ('text_positive', 'text_negative',)
    FUNCTION = 'prompt_styler'
    CATEGORY = 'utils'

    def prompt_styler(self, text_positive, text_negative, style, log_prompt):
        # Process and combine prompts in templates
        # The function replaces the positive prompt placeholder in the template,
        # and combines the negative prompt with the template's negative prompt, if they exist.
        text_positive_styled, text_negative_styled = read_sdxl_templates_replace_and_combine(self.json_data, style,
                                                                                             text_positive,
                                                                                             text_negative)

        # If logging is enabled (log_prompt is set to "Yes"), 
        # print the style, positive and negative text, and positive and negative prompts to the console
        if log_prompt == "Yes":
            print(f"style: {style}")
            print(f"text_positive: {text_positive}")
            print(f"text_negative: {text_negative}")
            print(f"text_positive_styled: {text_positive_styled}")
            print(f"text_negative_styled: {text_negative_styled}")

        return text_positive_styled, text_negative_styled

```
