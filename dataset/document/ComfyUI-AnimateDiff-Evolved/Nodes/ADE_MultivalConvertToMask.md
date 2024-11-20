# Multival to Mask 🎭🅐🅓
## Documentation
- Class name: `ADE_MultivalConvertToMask`
- Category: `Animate Diff 🎭🅐🅓/multival`
- Output node: `False`

The ADE_MultivalConvertToMask node is designed to transform multivalued inputs into mask format, accommodating both single float values and tensors. This conversion facilitates the use of multivalued data as masks in various processing pipelines, ensuring compatibility and extending the utility of input data by converting it into a standardized mask format.
## Input types
### Required
- **`multival`**
    - The 'multival' input accepts either a single float value or a tensor, representing the multivalued data to be converted into a mask. This flexibility allows for a wide range of data types to be seamlessly integrated into mask-based workflows.
    - Comfy dtype: `MULTIVAL`
    - Python dtype: `Union[float, torch.Tensor]`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output is a mask, either directly utilizing the input tensor if already in an appropriate format, or creating a new mask with the specified value if a float is provided. This ensures that the output is always in a mask format, ready for further processing.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MultivalConvertToMaskNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "multival": ("MULTIVAL",)
            }
        }
    
    RETURN_TYPES = ("MASK",)
    CATEGORY = "Animate Diff 🎭🅐🅓/multival"
    FUNCTION = "convert_multival_to_mask"

    def convert_multival_to_mask(self, multival: Union[float, Tensor]):
        # if already tensor, assume is a valid mask
        if type(multival) == Tensor:
            return (multival,)
        # otherwise, make a single 1x1 mask with the proper value
        shape = (1,1,1)
        converted_multival = torch.ones(shape) * multival
        return (converted_multival,)

```
