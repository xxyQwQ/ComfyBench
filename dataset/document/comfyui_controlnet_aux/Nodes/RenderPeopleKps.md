---
tags:
- Animation
- PoseEstimation
---

# Render Pose JSON (Human)
## Documentation
- Class name: `RenderPeopleKps`
- Category: `ControlNet Preprocessors/Pose Keypoint Postprocess`
- Output node: `False`

This node is designed for rendering human pose keypoints into images. It processes pose keypoints data, applying options to render body, hand, and face details, and outputs the rendered image as a numpy array.
## Input types
### Required
- **`kps`**
    - The pose keypoints data to be rendered. It is crucial for determining the positions and orientations of different body parts in the rendered image.
    - Comfy dtype: `POSE_KEYPOINT`
    - Python dtype: `list or tuple`
- **`render_body`**
    - A boolean flag to indicate whether to render the body keypoints. It affects the visual representation of the body in the output image.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`render_hand`**
    - A boolean flag to indicate whether to render the hand keypoints. It influences the inclusion of hand details in the rendered image.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`render_face`**
    - A boolean flag to indicate whether to render the face keypoints. It determines the depiction of facial features in the output image.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a rendered image of human pose keypoints, represented as a numpy array.
    - Python dtype: `np.ndarray`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RenderPeopleKps:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "kps": ("POSE_KEYPOINT",),
                "render_body": ("BOOLEAN", {"default": True}),
                "render_hand": ("BOOLEAN", {"default": True}),
                "render_face": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "ControlNet Preprocessors/Pose Keypoint Postprocess"

    def render(self, kps, render_body, render_hand, render_face) -> tuple[np.ndarray]:
        if isinstance(kps, list):
            kps = kps[0]

        poses, _, height, width = decode_json_as_poses(kps)
        np_image = draw_poses(
            poses,
            height,
            width,
            render_body,
            render_hand,
            render_face,
        )
        return (numpy2torch(np_image),)

```
