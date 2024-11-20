---
tags:
- Face
- FaceRestoration
---

# ReActor 🌌 Face Booster
## Documentation
- Class name: `ReActorFaceBoost`
- Category: `🌌 ReActor`
- Output node: `False`

The ReActorFaceBoost node is designed to enhance the quality of facial images through advanced restoration techniques. It leverages deep learning models to upscale, denoise, and improve the overall visual fidelity of faces in images, making it suitable for applications requiring high-quality facial visuals.
## Input types
### Required
- **`enabled`**
    - Activates or deactivates the face restoration process, allowing users to control when face enhancement is applied.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`boost_model`**
    - Specifies the deep learning model used for face restoration, affecting the quality and style of the output. Different models can be chosen based on desired resolution and restoration features.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`interpolation`**
    - Determines the method used for resizing images during the restoration process, impacting the smoothness and clarity of the upscaled face.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`visibility`**
    - Controls the blend between the original and restored face, allowing for fine-tuning of the restoration effect to achieve a natural-looking result.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`codeformer_weight`**
    - Adjusts the influence of the CodeFormer model in the restoration process, allowing for customization of the restoration effect based on the model's weight.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`restore_with_main_after`**
    - Determines whether additional restoration steps should be applied after the main restoration process, enabling further enhancement of the facial image.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`face_boost`**
    - Comfy dtype: `FACE_BOOST`
    - The enhanced facial image, with improvements in clarity, noise reduction, and resolution, as a result of the restoration process.
    - Python dtype: `numpy.ndarray`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
# Built-in or C extension class, unable to automatically detect source code
```
