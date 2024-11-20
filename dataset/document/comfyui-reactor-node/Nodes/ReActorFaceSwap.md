---
tags:
- Face
- ReActorFace
---

# ReActor 🌌 Fast Face Swap
## Documentation
- Class name: `ReActorFaceSwap`
- Category: `🌌 ReActor`
- Output node: `False`

The ReActorFaceSwap node specializes in performing fast face swaps within images, leveraging advanced algorithms to seamlessly replace faces in a given image with those from another source. This node is designed to handle the complexities of face swapping, including alignment, scaling, and blending, to produce natural-looking results.
## Input types
### Required
- **`enabled`**
    - Indicates if the face swap operation is currently enabled, controlling whether the swapping process should proceed.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`input_image`**
    - The image that serves as either the source or target for the face swap operation, depending on the context of the swap.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image.Image`
- **`swap_model`**
    - Specifies the model used for the face swapping process, determining the algorithm and parameters for the swap.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`facedetection`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`face_restore_model`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`face_restore_visibility`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`codeformer_weight`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`detect_gender_input`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`detect_gender_source`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`input_faces_index`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
- **`source_faces_index`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
- **`console_log_level`**
    - unknown
    - Comfy dtype: `COMBO[INT]`
    - Python dtype: `unknown`
### Optional
- **`source_image`**
    - unknown
    - Comfy dtype: `IMAGE`
    - Python dtype: `unknown`
- **`face_model`**
    - unknown
    - Comfy dtype: `FACE_MODEL`
    - Python dtype: `unknown`
- **`face_boost`**
    - unknown
    - Comfy dtype: `FACE_BOOST`
    - Python dtype: `unknown`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The final image after the face swap operation has been completed, showcasing the seamless integration of the swapped face.
    - Python dtype: `Image.Image`
- **`face_model`**
    - Comfy dtype: `FACE_MODEL`
    - The model used for the face swap, providing details about the algorithm and parameters applied.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)
    - [UltimateSDUpscale](../../ComfyUI_UltimateSDUpscale/Nodes/UltimateSDUpscale.md)
    - [VAEEncodeTiled](../../Comfy/Nodes/VAEEncodeTiled.md)
    - [FaceDetailer](../../ComfyUI-Impact-Pack/Nodes/FaceDetailer.md)
    - [UltimateSDUpscaleNoUpscale](../../ComfyUI_UltimateSDUpscale/Nodes/UltimateSDUpscaleNoUpscale.md)
    - SaveVideo
    - [VHS_VideoCombine](../../ComfyUI-VideoHelperSuite/Nodes/VHS_VideoCombine.md)
    - [Image Lucy Sharpen](../../was-node-suite-comfyui/Nodes/Image Lucy Sharpen.md)
    - [ImpactMakeImageList](../../ComfyUI-Impact-Pack/Nodes/ImpactMakeImageList.md)
    - [ImageScaleBy](../../Comfy/Nodes/ImageScaleBy.md)



## Source code
```python
# Built-in or C extension class, unable to automatically detect source code
```
