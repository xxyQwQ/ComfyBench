# Face Warp
## Documentation
- Class name: `FaceWarp`
- Category: `FaceAnalysis`
- Output node: `False`

The FaceWarp node is designed to perform facial warping between two images by utilizing facial landmarks to align features from a source image to a target image. This process involves calculating a transformation matrix based on the landmarks, applying the transformation to warp the source image and its mask to match the target image's geometry, and blending the images for a seamless transition.
## Input types
### Required
- **`analysis_models`**
    - A collection of models used for analyzing facial features, including landmark detection and other preprocessing tasks. It plays a crucial role in determining the transformation matrix for warping.
    - Comfy dtype: `ANALYSIS_MODELS`
    - Python dtype: `dict`
- **`image_from`**
    - The source image from which facial features will be warped.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image_to`**
    - The target image to which the source image's facial features will be aligned.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`keypoints`**
    - Specifies the set of facial landmarks to be used for the warping process, influencing the precision and areas of alignment.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list[str]`
- **`grow`**
    - A parameter that controls the expansion of the mask around the detected facial landmarks, affecting the area of the image to be warped.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`blur`**
    - Determines the level of blurring applied to the edges of the warped image and mask, enhancing the blending effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting image after applying the warping and blending processes, where the source image's features have been aligned to the target image.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The mask generated during the warping process, indicating the areas of the source image that have been transformed and blended into the target image.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class FaceWarp:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS", ),
                "image_from": ("IMAGE", ),
                "image_to": ("IMAGE", ),
                "keypoints": (["main features", "full face", "full face+forehead (if available)"], ),
                "grow": ("INT", { "default": 0, "min": -4096, "max": 4096, "step": 1 }),
                "blur": ("INT", { "default": 13, "min": 1, "max": 4096, "step": 2 }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "warp"
    CATEGORY = "FaceAnalysis"

    def warp(self, analysis_models, image_from, image_to, keypoints, grow, blur):
        import cv2
        from color_matcher import ColorMatcher
        from color_matcher.normalizer import Normalizer

        cm = ColorMatcher()
        image_from = tensor_to_image(image_from[0])
        image_to = tensor_to_image(image_to[0])

        shape_from = analysis_models.get_landmarks(image_from, extended_landmarks=("forehead" in keypoints))
        shape_to = analysis_models.get_landmarks(image_to, extended_landmarks=("forehead" in keypoints))

        if keypoints == "main features":
            shape_from = shape_from[1]
            shape_to = shape_to[1]
        else:
            shape_from = shape_from[0]
            shape_to = shape_to[0]

        # get the transformation matrix
        from_points = np.array(shape_from, dtype=np.float64)
        to_points = np.array(shape_to, dtype=np.float64)
        
        matrix = cv2.estimateAffine2D(from_points, to_points)[0]
        output = cv2.warpAffine(image_from, matrix, (image_to.shape[1], image_to.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        mask_from = mask_from_landmarks(image_from, shape_from)
        mask_to = mask_from_landmarks(image_to, shape_to)
        output_mask = cv2.warpAffine(mask_from, matrix, (image_to.shape[1], image_to.shape[0]))

        output_mask = torch.from_numpy(output_mask).unsqueeze(0).unsqueeze(-1).float()
        mask_to = torch.from_numpy(mask_to).unsqueeze(0).unsqueeze(-1).float()
        output_mask = torch.min(output_mask, mask_to)

        output = image_to_tensor(output).unsqueeze(0)
        image_to = image_to_tensor(image_to).unsqueeze(0)
        
        if grow != 0:
            output_mask = expand_mask(output_mask.squeeze(-1), grow, True).unsqueeze(-1)

        if blur > 1:
            if blur % 2 == 0:
                blur+= 1
            output_mask = T.functional.gaussian_blur(output_mask.permute(0,3,1,2), blur).permute(0,2,3,1)

        padding = 0

        _, y, x, _ = torch.where(mask_to)
        x1 = max(0, x.min().item() - padding)
        y1 = max(0, y.min().item() - padding)
        x2 = min(image_to.shape[2], x.max().item() + padding)
        y2 = min(image_to.shape[1], y.max().item() + padding)
        cm_ref = image_to[:, y1:y2, x1:x2, :]

        _, y, x, _ = torch.where(output_mask)
        x1 = max(0, x.min().item() - padding)
        y1 = max(0, y.min().item() - padding)
        x2 = min(output.shape[2], x.max().item() + padding)
        y2 = min(output.shape[1], y.max().item() + padding)
        cm_image = output[:, y1:y2, x1:x2, :]

        normalized = cm.transfer(src=Normalizer(cm_image[0].numpy()).type_norm() , ref=Normalizer(cm_ref[0].numpy()).type_norm(), method='mkl')
        normalized = torch.from_numpy(normalized).unsqueeze(0)

        factor = 0.8

        output[:, y1:y1+cm_image.shape[1], x1:x1+cm_image.shape[2], :] = factor * normalized + (1 - factor) * cm_image

        output_image = output * output_mask + image_to * (1 - output_mask)
        output_image = output_image.clamp(0, 1)
        output_mask = output_mask.clamp(0, 1).squeeze(-1)

        return (output_image, output_mask)

```
