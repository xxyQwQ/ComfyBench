---
tags:
- BoundingBox
- Image
- ImageTransformation
---

# Split Bboxes
## Documentation
- Class name: `SplitBboxes`
- Category: `KJNodes/masking`
- Output node: `False`

Splits a list of bounding boxes (bboxes) at a specified index into two separate lists, facilitating operations that require segmenting bounding box data.
## Input types
### Required
- **`bboxes`**
    - The list of bounding boxes to be split. This parameter is crucial for determining how the bounding boxes are divided into two groups.
    - Comfy dtype: `BBOX`
    - Python dtype: `Tuple[BBOX]`
- **`index`**
    - The index at which the list of bounding boxes is split. This parameter dictates the division point, affecting the composition of the resulting bounding box lists.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`bboxes_a`**
    - Comfy dtype: `BBOX`
    - The first sublist of bounding boxes, containing elements from the start of the original list up to (but not including) the specified index.
    - Python dtype: `Tuple[BBOX]`
- **`bboxes_b`**
    - Comfy dtype: `BBOX`
    - The second sublist of bounding boxes, containing elements from the specified index to the end of the original list.
    - Python dtype: `Tuple[BBOX]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SplitBboxes:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "index": ("INT", {"default": 0,"min": 0, "max": 99999999, "step": 1}),
            },
        }

    RETURN_TYPES = ("BBOX","BBOX",)
    RETURN_NAMES = ("bboxes_a","bboxes_b",)
    FUNCTION = "splitbbox"
    CATEGORY = "KJNodes/masking"
    DESCRIPTION = """
Splits the specified bbox list at the given index into two lists.
"""

    def splitbbox(self, bboxes, index):
        bboxes_a = bboxes[:index]  # Sub-list from the start of bboxes up to (but not including) the index
        bboxes_b = bboxes[index:]  # Sub-list from the index to the end of bboxes

        return (bboxes_a, bboxes_b,)

```
