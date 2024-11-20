---
tags:
- Prompt
- PromptStyling
---

# Styles Selector
## Documentation
- Class name: `easy stylesSelector`
- Category: `EasyUse/Prompt`
- Output node: `True`

The `easy stylesSelector` node is designed to facilitate the selection of styles for prompts in a user-friendly manner. It abstracts the complexity of choosing and applying styles by providing a simplified interface for users to select from predefined style options, enhancing the customization and creativity of prompt generation.
## Input types
### Required
- **`styles`**
    - This parameter allows users to select from a list of predefined styles, determining the visual and thematic presentation of the prompt.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`positive`**
    - The positive prompt input where users can specify custom text to influence the style selection process positively.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`negative`**
    - The negative prompt input where users can specify custom text to influence the style selection process negatively.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`positive`**
    - Comfy dtype: `STRING`
    - The positive styled prompt, reflecting the user's style selection and customizations.
    - Python dtype: `str`
- **`negative`**
    - Comfy dtype: `STRING`
    - The negative styled prompt, reflecting the user's style selection and customizations.
    - Python dtype: `str`
- **`ui`**
    - Indicates whether the selected styles are linked to the prompt, providing a boolean value that signifies the presence of a styles selection for the given prompt and user context.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class stylesPromptSelector:

    @classmethod
    def INPUT_TYPES(s):
        styles = ["fooocus_styles"]
        styles_dir = FOOOCUS_STYLES_DIR
        for file_name in os.listdir(styles_dir):
            file = os.path.join(styles_dir, file_name)
            if os.path.isfile(file) and file_name.endswith(".json") and "styles" in file_name.split(".")[0]:
                styles.append(file_name.split(".")[0])
        return {
            "required": {
               "styles": (styles, {"default": "fooocus_styles"}),
            },
            "optional": {
                "positive": ("STRING", {"forceInput": True}),
                "negative": ("STRING", {"forceInput": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("positive", "negative",)

    CATEGORY = 'EasyUse/Prompt'
    FUNCTION = 'run'
    OUTPUT_NODE = True

    def run(self, styles, positive='', negative='', prompt=None, extra_pnginfo=None, my_unique_id=None):
        values = []
        all_styles = {}
        positive_prompt, negative_prompt = '', negative
        if styles == "fooocus_styles":
            file = os.path.join(RESOURCES_DIR,  styles + '.json')
        else:
            file = os.path.join(FOOOCUS_STYLES_DIR, styles + '.json')
        f = open(file, 'r', encoding='utf-8')
        data = json.load(f)
        f.close()
        for d in data:
            all_styles[d['name']] = d
        if my_unique_id in prompt:
            if prompt[my_unique_id]["inputs"]['select_styles']:
                values = prompt[my_unique_id]["inputs"]['select_styles'].split(',')

        has_prompt = False
        if len(values) == 0:
            return (positive, negative)

        for index, val in enumerate(values):
            if 'prompt' in all_styles[val]:
                if "{prompt}" in all_styles[val]['prompt'] and has_prompt == False:
                    positive_prompt = all_styles[val]['prompt'].format(prompt=positive)
                    has_prompt = True
                else:
                    positive_prompt += ', ' + all_styles[val]['prompt'].replace(', {prompt}', '').replace('{prompt}', '')
            if 'negative_prompt' in all_styles[val]:
                negative_prompt += ', ' + all_styles[val]['negative_prompt'] if negative_prompt else all_styles[val]['negative_prompt']

        if has_prompt == False and positive:
            positive_prompt = positive + ', '

        return (positive_prompt, negative_prompt)

```
