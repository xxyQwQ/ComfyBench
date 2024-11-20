# Face Align
## Documentation
- Class name: `FaceAlign`
- Category: `FaceAnalysis`
- Output node: `False`

The FaceAlign node is designed for aligning faces within images based on facial keypoints. It adjusts the orientation of a face in an input image to match the orientation of a face in a target image or to a standard alignment if no target is provided, enhancing the consistency of facial analysis or recognition tasks.
## Input types
### Required
- **`analysis_models`**
    - Specifies the models used for facial analysis, particularly for detecting facial keypoints. Its role is crucial in determining the orientation and alignment of faces within images.
    - Comfy dtype: `ANALYSIS_MODELS`
    - Python dtype: `dict`
- **`image_from`**
    - The source image containing the face to be aligned. This image is processed to detect facial keypoints and adjust its orientation based on these points.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
### Optional
- **`image_to`**
    - An optional target image used to align the source image's face orientation with the target's face orientation. If provided, the alignment is adjusted to match the target face's orientation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The aligned image with the face orientation adjusted either to match the target image or to a standard alignment if no target is provided.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class FaceAlign:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS", ),
                "image_from": ("IMAGE", ),
            }, "optional": {
                "image_to": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "align"
    CATEGORY = "FaceAnalysis"

    def align(self, analysis_models, image_from, image_to=None):
        image_from = tensor_to_image(image_from[0])
        shape = analysis_models.get_keypoints(image_from)
        l_eye_from = shape[0]
        r_eye_from = shape[1]
        angle = float(np.degrees(np.arctan2(l_eye_from[1] - r_eye_from[1], l_eye_from[0] - r_eye_from[0])))

        if image_to is not None:
            image_to = tensor_to_image(image_to[0])
            shape = analysis_models.get_keypoints(image_to)
            l_eye_to = shape[0]
            r_eye_to = shape[1]
            angle -= float(np.degrees(np.arctan2(l_eye_to[1] - r_eye_to[1], l_eye_to[0] - r_eye_to[0])))

        # rotate the image
        image_from = Image.fromarray(image_from).rotate(angle)
        image_from = image_to_tensor(image_from).unsqueeze(0)

        #img = np.array(Image.fromarray(image_from).rotate(angle))
        #img = image_to_tensor(img).unsqueeze(0)

        return (image_from, )

```
