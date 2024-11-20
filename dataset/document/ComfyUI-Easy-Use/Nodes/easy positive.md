# Positive
## Documentation
- Class name: `easy positive`
- Category: `EasyUse/Prompt`
- Output node: `False`

The 'easy positive' node is designed to process and enhance positive textual inputs for generative tasks, utilizing advanced encoding techniques and conditional embeddings to optimize the generation process. It integrates seamlessly with language models and image processing pipelines, emphasizing the positive aspects of inputs to influence the output generation towards more favorable outcomes.
## Input types
### Required
- **`positive`**
    - Represents the positive textual input that the node processes. It is crucial for defining the desired positive attributes or themes that should be emphasized in the output generation, directly influencing the nature and direction of the generative task.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`positive`**
    - Comfy dtype: `STRING`
    - The enhanced version of the input text, optimized for positive influence on the generative process. This output is the result of advanced encoding and processing, ready to be used in further generative tasks.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class positivePrompt:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "positive": ("STRING", {"default": "", "multiline": True, "placeholder": "Positive"}),}
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "main"

    CATEGORY = "EasyUse/Prompt"

    @staticmethod
    def main(positive):
        if has_chinese(positive):
            return zh_to_en([positive])
        return positive,

```
