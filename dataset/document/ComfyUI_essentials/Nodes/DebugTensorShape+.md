# 🔧 Debug Tensor Shape
## Documentation
- Class name: `DebugTensorShape+`
- Category: `essentials/utilities`
- Output node: `True`

The DebugTensorShape node is designed for debugging purposes, specifically to inspect and print the shapes of tensors within a data structure. It recursively traverses through nested tensors, including those within dictionaries and lists, to collect and display their shapes, aiding in the understanding of data flow and structure in tensor-based computations.
## Input types
### Required
- **`tensor`**
    - The 'tensor' parameter is the primary input for the DebugTensorShape node. Its purpose is to enable the inspection of tensor shapes within complex data structures, facilitating debugging and analysis.
    - Comfy dtype: `*`
    - Python dtype: `Union[torch.Tensor, Dict, List]`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class DebugTensorShape:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (any, {}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "essentials/utilities"
    OUTPUT_NODE = True

    def execute(self, tensor):
        shapes = []
        def tensorShape(tensor):
            if isinstance(tensor, dict):
                for k in tensor:
                    tensorShape(tensor[k])
            elif isinstance(tensor, list):
                for i in range(len(tensor)):
                    tensorShape(tensor[i])
            elif hasattr(tensor, 'shape'):
                shapes.append(list(tensor.shape))

        tensorShape(tensor)

        print(f"\033[96mShapes found: {shapes}\033[0m")

        return (None,)

```
