---
tags:
- IdentityImage
---

# Easy Apply PuLID
## Documentation
- Class name: `easy pulIDApply`
- Category: `EasyUse/Adapter`
- Output node: `False`

This node facilitates the application of the PulID model to images, enhancing them based on predefined or custom parameters. It integrates with other models like InsightFace for face analysis and Eva Clip for additional processing, aiming to improve image quality or modify images according to specific criteria.
## Input types
### Required
- **`model`**
    - The base model to which the PulID enhancements will be applied. It serves as the foundation for the image transformation process.
    - Comfy dtype: `MODEL`
    - Python dtype: `MODEL`
- **`pulid_file`**
    - Specifies the file path to the PulID model to be used for the image enhancement. This parameter allows for the dynamic selection of different PulID models.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`insightface`**
    - Defines the InsightFace model configuration to be used for face analysis within the image. This can be CPU, CUDA, or ROCM based, affecting the performance and accuracy of face detection.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`image`**
    - The image to be processed and enhanced by the PulID model. It is the primary input over which the transformations are applied.
    - Comfy dtype: `IMAGE`
    - Python dtype: `IMAGE`
- **`method`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`weight`**
    - A floating-point value that adjusts the intensity of the PulID model's effect on the image. It allows for fine-tuning the strength of the enhancements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`start_at`**
    - Defines the starting point of the effect application in terms of image processing, allowing for partial enhancements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_at`**
    - Specifies the endpoint for the PulID model's effect application, enabling selective enhancement within the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`attn_mask`**
    - An optional mask that can be applied to focus the PulID model's enhancements on specific areas of the image, providing more control over the output.
    - Comfy dtype: `MASK`
    - Python dtype: `MASK`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The enhanced model after applying the PulID transformations. It represents the output of the image enhancement process.
    - Python dtype: `MODEL`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class applyPulID:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "pulid_file": (folder_paths.get_filename_list("pulid"),),
                "insightface": (["CPU", "CUDA", "ROCM"],),
                "image": ("IMAGE",),
                "method": (["fidelity", "style", "neutral"],),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "attn_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    FUNCTION = "run"
    CATEGORY = "EasyUse/Adapter"

    def error(self):
        raise Exception(f"[ERROR] To use pulIDApply, you need to install 'ComfyUI_PulID'")

    def run(self, model, image, pulid_file, insightface, weight, start_at, end_at, method=None, noise=0.0, fidelity=None, projection=None, attn_mask=None, prompt=None, extra_pnginfo=None, my_unique_id=None):
        pulid_model, insightface_model, eva_clip = None, None, None
        # Load PulID
        cache_key = 'pulID'
        if cache_key in backend_cache.cache:
            log_node_info("easy pulIDApply","Using InstantIDModel Cached")
            _, pulid_model = backend_cache.cache[cache_key][1]
        if "PulidModelLoader" in ALL_NODE_CLASS_MAPPINGS:
            load_pulid_cls = ALL_NODE_CLASS_MAPPINGS["PulidModelLoader"]
            pulid_model, = load_pulid_cls().load_model(pulid_file)
            backend_cache.update_cache(cache_key, 'pulid', (False, pulid_model))
        else:
            self.error()
        # Load Insightface
        icache_key = 'insightface-' + insightface
        if icache_key in backend_cache.cache:
            log_node_info("easy pulIDApply", f"Using InsightFaceModel {insightface} Cached")
            _, insightface_model = backend_cache.cache[icache_key][1]
        elif "PulidInsightFaceLoader" in ALL_NODE_CLASS_MAPPINGS:
            load_insightface_cls = ALL_NODE_CLASS_MAPPINGS["InstantIDFaceAnalysis"]
            insightface_model, = load_insightface_cls().load_insight_face(insightface)
            backend_cache.update_cache(icache_key, 'insightface', (False, insightface_model))
        else:
            self.error()
        # Load Eva clip
        ecache_key = 'eva_clip'
        if ecache_key in backend_cache.cache:
            log_node_info("easy pulIDApply", f"Using EVAClipModel Cached")
            _, eva_clip = backend_cache.cache[ecache_key][1]
        elif "PulidEvaClipLoader" in ALL_NODE_CLASS_MAPPINGS:
            load_evaclip_cls = ALL_NODE_CLASS_MAPPINGS["PulidEvaClipLoader"]
            eva_clip, = load_evaclip_cls().load_eva_clip()
            backend_cache.update_cache(ecache_key, 'eva_clip', (False, eva_clip))
        else:
            self.error()

        # Apply PulID
        if method is not None:
            if "ApplyPulid" in ALL_NODE_CLASS_MAPPINGS:
                cls = ALL_NODE_CLASS_MAPPINGS['ApplyPulid']
                model, = cls().apply_pulid(model, pulid=pulid_model, eva_clip=eva_clip, face_analysis=insightface_model, image=image, weight=weight, method=method, start_at=start_at, end_at=end_at, attn_mask=attn_mask)
            else:
                self.error()
        else:
            if "ApplyPulidAdvanced" in ALL_NODE_CLASS_MAPPINGS:
                cls = ALL_NODE_CLASS_MAPPINGS['ApplyPulidAdvanced']
                model, = cls().apply_pulid(model, pulid=pulid_model, eva_clip=eva_clip, face_analysis=insightface_model, image=image, weight=weight, projection=projection, fidelity=fidelity, noise=noise, start_at=start_at, end_at=end_at, attn_mask=attn_mask)
            else:
                self.error()

        return (model,)

```
