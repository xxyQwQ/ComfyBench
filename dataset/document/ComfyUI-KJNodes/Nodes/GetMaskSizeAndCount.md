# Get Mask Size & Count
## Documentation
- Class name: `GetMaskSizeAndCount`
- Category: `KJNodes/masking`
- Output node: `False`

This node is designed to analyze the dimensions and batch size of a given mask, returning its width, height, and count while passing the mask through unchanged. It serves as a utility for understanding and processing mask dimensions within a workflow.
## Input types
### Required
- **`mask`**
    - The mask input is crucial for determining the dimensions and batch size of the mask. It directly influences the node's output by providing the necessary data for calculation.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The original mask passed through unchanged, facilitating further processing or analysis.
    - Python dtype: `torch.Tensor`
- **`width`**
    - Comfy dtype: `INT`
    - The width of the mask, providing insight into its dimensions.
    - Python dtype: `int`
- **`height`**
    - Comfy dtype: `INT`
    - The height of the mask, offering information on its size.
    - Python dtype: `int`
- **`count`**
    - Comfy dtype: `INT`
    - The batch size of the mask, indicating how many masks are processed.
    - Python dtype: `int`
- **`ui`**
    - A textual representation of the mask's dimensions and count, enhancing readability and understanding.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class GetMaskSizeAndCount:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mask": ("MASK",),
        }}

    RETURN_TYPES = ("MASK","INT", "INT", "INT",)
    RETURN_NAMES = ("mask", "width", "height", "count",)
    FUNCTION = "getsize"
    CATEGORY = "KJNodes/masking"
    DESCRIPTION = """
Returns the width, height and batch size of the mask,  
and passes it through unchanged.  

"""

    def getsize(self, mask):
        width = mask.shape[2]
        height = mask.shape[1]
        count = mask.shape[0]
        return {"ui": {
            "text": [f"{count}x{width}x{height}"]}, 
            "result": (mask, width, height, count) 
        }

```
