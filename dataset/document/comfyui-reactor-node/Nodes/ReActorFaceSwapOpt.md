---
tags:
- Face
- ReActorFace
---

# ReActor 🌌 Fast Face Swap [OPTIONS]
## Documentation
- Class name: `ReActorFaceSwapOpt`
- Category: `🌌 ReActor`
- Output node: `False`

The ReActorFaceSwapOpt node offers an enhanced face swapping functionality with additional options for customization. It leverages advanced techniques to swap faces in images, providing users with the ability to fine-tune the process for improved results.
## Input types
### Required
- **`enabled`**
    - Indicates if the face swapping functionality is currently enabled, controlling whether the operation should proceed.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`input_image`**
    - The image that will undergo the face swapping process, serving as the canvas for the operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image.Image`
- **`swap_model`**
    - Specifies the model used for the face swapping process, affecting the accuracy and quality of the swap.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`facedetection`**
    - Defines the method or model used for detecting faces within the images, crucial for identifying target faces for swapping.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`face_restore_model`**
    - The model used for restoring or enhancing the quality of faces post-swap, contributing to more realistic outcomes.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`face_restore_visibility`**
    - Controls the visibility level of the restored faces, allowing for fine-tuning of the final appearance.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`codeformer_weight`**
    - Determines the weight of the CodeFormer model in the face restoration process, influencing the balance between original and enhanced features.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`source_image`**
    - The source image providing the face to be swapped into the target image, key to the face swapping operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image.Image`
- **`face_model`**
    - Specifies the face model used for the swapping process, impacting the swap's accuracy and compatibility with different face types.
    - Comfy dtype: `FACE_MODEL`
    - Python dtype: `str`
- **`options`**
    - A collection of additional options that can be configured to customize the face swapping process.
    - Comfy dtype: `OPTIONS`
    - Python dtype: `Dict[str, Any]`
- **`face_boost`**
    - Indicates whether an additional face enhancement feature is enabled, aiming to improve the quality of the swapped face.
    - Comfy dtype: `FACE_BOOST`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The final image after the face swapping operation, showcasing the swapped and potentially enhanced face.
    - Python dtype: `Image.Image`
- **`face_model`**
    - Comfy dtype: `FACE_MODEL`
    - The face model used during the swapping process, potentially including modifications or enhancements applied.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
# Built-in or C extension class, unable to automatically detect source code
```
