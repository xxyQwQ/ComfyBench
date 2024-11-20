# Negative
## Documentation
- Class name: `easy negative`
- Category: `EasyUse/Prompt`
- Output node: `False`

This node specializes in processing negative textual inputs for generative models, focusing on embedding and conditioning these inputs to influence the generative process negatively. It integrates with CLIP models for encoding text into embeddings and applies various normalization and interpretation strategies to tailor the negative influence on the generated content.
## Input types
### Required
- **`negative`**
    - The negative textual input to be processed. It is crucial for defining the aspects or features to be minimized or avoided in the generated content, affecting the overall direction and quality of the output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`negative`**
    - Comfy dtype: `STRING`
    - The processed negative text, now encoded and conditioned for use in influencing the generative model. This output is ready to be integrated into the model's generative process.
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class negativePrompt:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "negative": ("STRING", {"default": "", "multiline": True, "placeholder": "Negative"}),}
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("negative",)
    FUNCTION = "main"

    CATEGORY = "EasyUse/Prompt"

    @staticmethod
    def main(negative):
        if has_chinese(negative):
            return zh_to_en([negative])
        else:
            return negative,

```
