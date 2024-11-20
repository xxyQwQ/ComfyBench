# 🔧 Load CLIPSeg Models
## Documentation
- Class name: `LoadCLIPSegModels+`
- Category: `essentials/segmentation`
- Output node: `False`

This node is designed to load the CLIPSeg models for image segmentation, specifically initializing and returning the CLIPSeg processor and model pre-trained on a specific dataset. It abstracts the complexity of model loading, providing an easy-to-use interface for obtaining the necessary components for CLIPSeg-based segmentation tasks.
## Input types
### Required
## Output types
- **`clip_seg`**
    - Comfy dtype: `CLIP_SEG`
    - The output is a tuple containing the CLIPSeg processor and model, ready for use in image segmentation tasks.
    - Python dtype: `Tuple[CLIPSegProcessor, CLIPSegForImageSegmentation]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class LoadCLIPSegModels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
        }

    RETURN_TYPES = ("CLIP_SEG",)
    FUNCTION = "execute"
    CATEGORY = "essentials/segmentation"

    def execute(self):
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

        return ((processor, model),)

```
