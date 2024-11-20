---
tags:
- LLM
- LLMChat
---

# ∞ Get Prompt JSON
## Documentation
- Class name: `LLMGetPromptJSON`
- Category: `SALT/Language Toolkit/Misc`
- Output node: `False`

The LLMGetPromptJSON node is designed to transform a given prompt into a structured JSON format, optionally parsing and including various input types such as integers, floats, and booleans. This functionality facilitates the dynamic generation of prompts tailored to specific requirements, enhancing the flexibility and precision of language model interactions.
## Input types
### Required
### Optional
- **`optional_order_input`**
    - An optional parameter that allows for the specification of the order in which inputs are processed. This can affect the final structure and content of the generated JSON prompt.
    - Comfy dtype: `*`
    - Python dtype: `dict`
- **`parse_to_inputs`**
    - A boolean flag that, when set to true, triggers the parsing of the prompt into distinct inputs based on specified inclusion criteria (integers, floats, booleans).
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`include_int`**
    - Determines whether integers found within the prompt should be included in the parsed inputs, contributing to the structured JSON output.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`include_float`**
    - Specifies whether floating-point numbers within the prompt are included in the parsed inputs, affecting the composition of the JSON structure.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`include_boolean`**
    - Indicates whether boolean values within the prompt should be parsed and included in the JSON output, influencing the prompt's final structure.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`prompt_json`**
    - Comfy dtype: `STRING`
    - The resulting JSON-formatted prompt, structured according to the specified input parameters and parsing options.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMGetPromptJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "optional_order_input": (WILDCARD, {}),
                "parse_to_inputs": ("BOOLEAN", {"default": False}),
                "include_int": ("BOOLEAN", {"default": True}),
                "include_float": ("BOOLEAN", {"default": True}),
                "include_boolean": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT"
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_json",)

    FUNCTION = "get_prompt"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Misc"

    def get_prompt(self, prompt, optional_order_input=None, parse_to_inputs=False, include_int=True, include_float=True, include_boolean=True):
        if parse_to_inputs:
            prompt = extract_inputs(prompt, include_int, include_float, include_boolean)
        return (json.dumps(prompt), )
    
    @classmethod
    def IS_CHANGED(**kwargs):
        return float("NaN")

```
