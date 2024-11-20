# Render Pose JSON (Animal)
## Documentation
- Class name: `RenderAnimalKps`
- Category: `ControlNet Preprocessors/Pose Keypoint Postprocess`
- Output node: `False`

The RenderAnimalKps node is designed to process pose keypoints data for animals, converting it into a visual representation. It decodes the input keypoints, applies pose rendering to generate an image of the animal's pose, and returns the image as a numpy array.
## Input types
### Required
- **`kps`**
    - The 'kps' parameter represents the pose keypoints data for an animal. It is crucial for rendering the visual representation of the animal's pose, as it contains the spatial information needed to accurately depict the pose.
    - Comfy dtype: `POSE_KEYPOINT`
    - Python dtype: `list`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is an image representation of the animal's pose, provided as a numpy array.
    - Python dtype: `tuple[np.ndarray]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RenderAnimalKps:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "kps": ("POSE_KEYPOINT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "ControlNet Preprocessors/Pose Keypoint Postprocess"

    def render(self, kps) -> tuple[np.ndarray]:
        if isinstance(kps, list):
            kps = kps[0]

        _, poses, height, width = decode_json_as_poses(kps)
        np_image = draw_animalposes(poses, height, width)
        return (numpy2torch(np_image),)

```
