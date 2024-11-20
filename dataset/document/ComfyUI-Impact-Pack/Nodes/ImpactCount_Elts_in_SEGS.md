# Count Elts in SEGS
## Documentation
- Class name: `ImpactCount_Elts_in_SEGS`
- Category: `ImpactPack/Util`
- Output node: `False`

This node is designed to count the elements within a given SEGS data structure, providing a simple yet essential utility for quantifying the components of SEGS in the context of data processing or analysis.
## Input types
### Required
- **`segs`**
    - Represents the SEGS data structure whose elements are to be counted. It is crucial for determining the total number of elements contained within, impacting the node's output directly.
    - Comfy dtype: `SEGS`
    - Python dtype: `Tuple[Tuple[int, int], List[Any]]`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - The output is an integer representing the total number of elements within the provided SEGS data structure, serving as a quantitative measure of its content.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class Count_Elts_in_SEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "segs": ("SEGS", ),
                     },
                }

    RETURN_TYPES = ("INT",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, segs):
        return (len(segs[1]), )

```
