---
tags:
- IPAdapter
---

# ToIPAdapterPipe (Inspire)
## Documentation
- Class name: `ToIPAdapterPipe __Inspire`
- Category: `InspirePack/Util`
- Output node: `False`

The ToIPAdapterPipe node is designed to create a pipeline that integrates various components such as IP adapters, models, and optional vision and face recognition enhancements into a unified processing flow. This setup facilitates the adaptation and enhancement of input data or models for further processing or analysis.
## Input types
### Required
- **`ipadapter`**
    - The 'ipadapter' parameter is crucial for specifying the IP adapter component to be used in the pipeline, serving as the foundational element for data or model adaptation.
    - Comfy dtype: `IPADAPTER`
    - Python dtype: `str`
- **`model`**
    - The 'model' parameter identifies the specific model to be integrated into the pipeline, enabling its adaptation or enhancement through the IP adapter.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
### Optional
- **`clip_vision`**
    - The 'clip_vision' parameter optionally adds vision processing capabilities to the pipeline, leveraging CLIP models for enhanced visual understanding.
    - Comfy dtype: `CLIP_VISION`
    - Python dtype: `str`
- **`insightface`**
    - The 'insightface' parameter optionally incorporates face recognition technology into the pipeline, providing advanced facial analysis features.
    - Comfy dtype: `INSIGHTFACE`
    - Python dtype: `str`
## Output types
- **`ipadapter_pipe`**
    - Comfy dtype: `IPADAPTER_PIPE`
    - This output represents the assembled pipeline, encapsulating the integrated IP adapter, model, and any optional enhancements for vision and face recognition.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ToIPAdapterPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": ("IPADAPTER", ),
                "model": ("MODEL",),
            },
            "optional": {
                "clip_vision": ("CLIP_VISION",),
                "insightface": ("INSIGHTFACE",),
            }
        }

    RETURN_TYPES = ("IPADAPTER_PIPE",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Util"

    @staticmethod
    def doit(ipadapter, model, clip_vision, insightface=None):
        pipe = ipadapter, model, clip_vision, insightface, lambda x: x

        return (pipe,)

```
