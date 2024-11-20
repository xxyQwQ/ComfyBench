---
tags:
- SAM
---

# Get SAM Embedding
## Documentation
- Class name: `GetSAMEmbedding`
- Category: `Art Venture/Segmentation`
- Output node: `False`

The GetSAMEmbedding node is designed to extract semantic embeddings from images using a specific SAM model. It facilitates the transformation of visual content into a format that can be further processed or analyzed, emphasizing the extraction of meaningful features that represent the underlying content of the images.
## Input types
### Required
- **`sam_model`**
    - The SAM model used for generating embeddings from images. It plays a crucial role in determining the quality and relevance of the extracted features.
    - Comfy dtype: `AV_SAM_MODEL`
    - Python dtype: `custom type representing a SAM model instance`
- **`image`**
    - The input image from which embeddings are to be extracted. This image is processed by the SAM model to generate a semantic representation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `PIL.Image or torch.Tensor representing an image`
### Optional
- **`device_mode`**
    - Specifies the computational device (AUTO, Prefer GPU, CPU) for model inference, affecting performance and resource utilization.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`sam_embedding`**
    - Comfy dtype: `SAM_EMBEDDING`
    - The semantic embedding generated from the input image, representing its meaningful features in a condensed form.
    - Python dtype: `numpy.ndarray`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class GetSAMEmbedding:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam_model": ("AV_SAM_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {"device_mode": (["AUTO", "Prefer GPU", "CPU"],)},
        }

    RETURN_TYPES = ("SAM_EMBEDDING",)
    CATEGORY = "Art Venture/Segmentation"
    FUNCTION = "get_sam_embedding"

    def get_sam_embedding(self, image, sam_model, device_mode="AUTO"):
        device = gpu if device_mode != "CPU" else cpu
        sam_model.to(device)

        ensure_package("segment_anything")
        from segment_anything import SamPredictor

        try:
            predictor = SamPredictor(sam_model)
            image = tensor2pil(image)
            image = image.convert("RGB")
            image = np.array(image)
            predictor.set_image(image, "RGB")
            embedding = predictor.get_image_embedding().cpu().numpy()

            return (embedding,)
        finally:
            if device_mode == "AUTO":
                sam_model.to(cpu)

```
