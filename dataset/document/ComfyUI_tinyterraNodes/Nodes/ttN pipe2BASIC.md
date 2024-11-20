---
tags:
- Image
- Pipeline
- PipelineTransformation
---

# pipe > basic_pipe
## Documentation
- Class name: `ttN pipe2BASIC`
- Category: `🌏 tinyterra/pipe`
- Output node: `False`

The ttN pipe2BASIC node is designed to simplify and streamline the processing pipeline by converting a complex input pipeline into a basic form. This node extracts essential components from the input pipeline, focusing on core elements such as model, clip, vae, and conditioning parameters, to facilitate further processing or analysis.
## Input types
### Required
- **`pipe`**
    - The 'pipe' parameter represents the complex input pipeline that is to be simplified. It is essential for the operation as it contains all the data and settings that will be streamlined into a basic form.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `Dict[str, Any]`
## Output types
- **`basic_pipe`**
    - Comfy dtype: `BASIC_PIPE`
    - The 'basic_pipe' output contains the simplified version of the input pipeline, focusing on essential components for further processing.
    - Python dtype: `Tuple[Any]`
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - Returns the original input pipeline, allowing for reference or further modifications.
    - Python dtype: `Dict[str, Any]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ttN_pipe_2BASIC:
    version = '1.1.0'
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                },
            "hidden": {"ttNnodeVersion": ttN_pipe_2BASIC.version},
            }

    RETURN_TYPES = ("BASIC_PIPE", "PIPE_LINE",)
    RETURN_NAMES = ("basic_pipe", "pipe",)
    FUNCTION = "flush"

    CATEGORY = "🌏 tinyterra/pipe"
    
    def flush(self, pipe):
        basic_pipe = (pipe.get('model'), pipe.get('clip'), pipe.get('vae'), pipe.get('positive'), pipe.get('negative'))
        return (basic_pipe, pipe, )

```
