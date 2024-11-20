---
tags:
- Face
---

# Face Analysis Models
## Documentation
- Class name: `FaceAnalysisModels`
- Category: `FaceAnalysis`
- Output node: `False`

The FaceAnalysisModels node is designed to load and manage face analysis models based on specified libraries and providers. It dynamically supports different face analysis libraries and hardware acceleration options, facilitating the flexible integration of face analysis capabilities into applications.
## Input types
### Required
- **`library`**
    - Specifies the face analysis library to use. The choice of library affects the underlying model and algorithms for face analysis.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`provider`**
    - Determines the hardware acceleration provider for the face analysis model, enabling optimized performance on various platforms.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
## Output types
- **`analysis_models`**
    - Comfy dtype: `ANALYSIS_MODELS`
    - Returns an instance of the loaded face analysis model, ready for performing face analysis tasks.
    - Python dtype: `Tuple[Union[InsightFace, DLib]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class FaceAnalysisModels:
    @classmethod
    def INPUT_TYPES(s):
        libraries = []
        if IS_INSIGHTFACE_INSTALLED:
            libraries.append("insightface")
        if IS_DLIB_INSTALLED:
            libraries.append("dlib")

        return {"required": {
            "library": (libraries, ),
            "provider": (["CPU", "CUDA", "DirectML", "OpenVINO", "ROCM", "CoreML"], ),
        }}

    RETURN_TYPES = ("ANALYSIS_MODELS", )
    FUNCTION = "load_models"
    CATEGORY = "FaceAnalysis"

    def load_models(self, library, provider):
        out = {}

        if library == "insightface":
            out = InsightFace(provider)
        else:
            out = DLib()

        return (out, )

```
