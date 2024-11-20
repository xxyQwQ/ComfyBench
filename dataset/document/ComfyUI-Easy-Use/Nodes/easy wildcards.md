---
tags:
- Prompt
- Text
- Wildcard
---

# Wildcards
## Documentation
- Class name: `easy wildcards`
- Category: `EasyUse/Prompt`
- Output node: `True`

The 'easy wildcards' node is designed to manage and process wildcards within strings, offering functionalities to define, normalize, read, and replace wildcards. It supports dynamic content generation by allowing the insertion of predefined or random elements into strings based on wildcard patterns. This node facilitates the customization and variability of text outputs, making it a versatile tool for generating dynamic content in applications.
## Input types
### Required
- **`text`**
    - A string input that supports Lora Block Weight and wildcard functionality, allowing for dynamic content generation through the use of placeholders and predefined text patterns.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`Select to add LoRA`**
    - Allows the selection of a LoRA to be added to the text, enhancing the input with additional, customizable layers of text transformation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`Select to add Wildcard`**
    - Enables the selection of a wildcard to be incorporated into the text, facilitating dynamic substitutions and content variability.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`seed`**
    - An integer value used to seed random number generators for consistent wildcard replacements across different executions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`multiline_mode`**
    - A boolean flag indicating whether the input text should be processed in multiline mode, affecting how wildcards and LoRAs are applied.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`text`**
    - Comfy dtype: `STRING`
    - The original input text with wildcards and LoRAs processed and applied, reflecting the dynamic content generation.
    - Python dtype: `str`
- **`populated_text`**
    - Comfy dtype: `STRING`
    - The text after processing, showing the result of wildcard and LoRA substitutions, providing a version of the text enriched with dynamic elements.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class wildcardsPrompt:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        wildcard_list = get_wildcard_list()
        return {"required": {
            "text": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False, "placeholder": "(Support Lora Block Weight and wildcard)"}),
            "Select to add LoRA": (["Select the LoRA to add to the text"] + folder_paths.get_filename_list("loras"),),
            "Select to add Wildcard": (["Select the Wildcard to add to the text"] + wildcard_list,),
            "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
            "multiline_mode": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "populated_text")
    OUTPUT_IS_LIST = (True, True)
    OUTPUT_NODE = True
    FUNCTION = "main"

    CATEGORY = "EasyUse/Prompt"

    def translate(self, text):
        if has_chinese(text):
            return zh_to_en([text])[0]
        else:
            return text

    def main(self, *args, **kwargs):
        prompt = kwargs["prompt"] if "prompt" in kwargs else None
        seed = kwargs["seed"]

        # Clean loaded_objects
        if prompt:
            easyCache.update_loaded_objects(prompt)

        text = kwargs['text']
        if "multiline_mode" in kwargs and kwargs["multiline_mode"]:
            populated_text = []
            _text = []
            text = text.split("\n")
            for t in text:
                t = self.translate(t)
                _text.append(t)
                populated_text.append(process(t, seed))
            text = _text
        else:
            text = self.translate(text)
            populated_text = [process(text, seed)]
            text = [text]
        return {"ui": {"value": [seed]}, "result": (text, populated_text)}

```
