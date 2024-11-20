# HF Transformers Classifier Provider
## Documentation
- Class name: `ImpactHFTransformersClassifierProvider`
- Category: `ImpactPack/HuggingFace`
- Output node: `False`

Provides a classification pipeline for images using pre-trained models from Hugging Face Transformers. It allows selecting a model from a predefined list or specifying a custom model repository ID, and supports device mode configuration for performance optimization.
## Input types
### Required
- **`preset_repo_id`**
    - Specifies the pre-trained model to use for classification. It can be selected from a predefined list of model URLs or set to 'Manual repo id' for custom models, enabling flexibility in model choice.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `Union[List[str], str]`
- **`manual_repo_id`**
    - The repository ID of a custom model to be used for classification when 'Manual repo id' is selected. This allows for the use of models outside the predefined list.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`device_mode`**
    - Determines the device (CPU or GPU) on which the classification pipeline will run, optimizing for performance or resource availability.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
## Output types
- **`transformers_classifier`**
    - Comfy dtype: `TRANSFORMERS_CLASSIFIER`
    - The classification pipeline configured with the selected model, ready for image classification tasks.
    - Python dtype: `transformers.Pipeline`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class HF_TransformersClassifierProvider:
    @classmethod
    def INPUT_TYPES(s):
        global hf_transformer_model_urls
        return {"required": {
                        "preset_repo_id": (hf_transformer_model_urls + ['Manual repo id'],),
                        "manual_repo_id": ("STRING", {"multiline": False}),
                        "device_mode": (["AUTO", "Prefer GPU", "CPU"],),
                     },
                }

    RETURN_TYPES = ("TRANSFORMERS_CLASSIFIER",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/HuggingFace"

    def doit(self, preset_repo_id, manual_repo_id, device_mode):
        from transformers import pipeline

        if preset_repo_id == 'Manual repo id':
            url = manual_repo_id
        else:
            url = preset_repo_id

        if device_mode != 'CPU':
            device = comfy.model_management.get_torch_device()
        else:
            device = "cpu"

        classifier = pipeline('image-classification', model=url, device=device)

        return (classifier,)

```
