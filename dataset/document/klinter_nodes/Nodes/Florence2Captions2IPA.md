# Florence2Captions2IPA (klinter)
## Documentation
- Class name: `Florence2Captions2IPA`
- Category: `text`
- Output node: `False`

The Florence2Captions2IPA node is designed to format input captions into a standardized string format. It accepts captions in various forms, including single strings or lists, and processes them into a unified string representation. This functionality is essential for preparing text data for further processing or analysis, ensuring consistency in input format.
## Input types
### Required
- **`caption_input`**
    - The 'caption_input' parameter can accept captions in multiple formats, including single strings or lists of strings. It plays a crucial role in determining how the input text is formatted for further processing, directly impacting the node's output.
    - Comfy dtype: `STRING`
    - Python dtype: `Union[str, List[str]]`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The output is a formatted string, which may be a concatenation of multiple input strings or a single input string, depending on the input type. This standardized output is suitable for subsequent text processing or analysis tasks.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class Florence2Captions2IPA:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "caption_input": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "format_prompts"
    CATEGORY = "text"

    def format_prompts(self, caption_input):
        if isinstance(caption_input, list):
            # If it's already a list, join the items
            formatted = "\n".join(caption_input)
        elif isinstance(caption_input, str):
            # If it's a single string, use it as is
            formatted = caption_input
        else:
            # If it's neither a list nor a string, convert to string
            formatted = str(caption_input)

        return (formatted,) 

```
