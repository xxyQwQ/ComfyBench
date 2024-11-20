# LayerUtility: TextJoin
## Documentation
- Class name: `LayerUtility: TextJoin`
- Category: `😺dzNodes/LayerUtility/Data`
- Output node: `False`

The TextJoin node is designed to concatenate multiple text inputs into a single string, separated by commas. It allows for the dynamic combination of up to four text strings, facilitating the aggregation of textual data for further processing or display.
## Input types
### Required
- **`text_i`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
### Optional
## Output types
- **`text`**
    - Comfy dtype: `STRING`
    - The output is a single string that represents the concatenated result of the input texts, separated by commas.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class TextJoin:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_1": ("STRING", {"multiline": False}),

            },
            "optional": {
                "text_2": ("STRING", {"multiline": False}),
                "text_3": ("STRING", {"multiline": False}),
                "text_4": ("STRING", {"multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "text_join"
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def text_join(self, **kwargs):

        texts = [kwargs[key] for key in kwargs if key.startswith('text')]
        combined_text = ', '.join(texts)
        return (combined_text,)

```
