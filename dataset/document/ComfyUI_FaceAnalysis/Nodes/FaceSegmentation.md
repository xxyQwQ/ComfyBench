---
tags:
- Segmentation
---

# Face Segmentation
## Documentation
- Class name: `FaceSegmentation`
- Category: `FaceAnalysis`
- Output node: `False`

The FaceSegmentation node is designed to process images by segmenting faces based on provided parameters. It utilizes advanced image processing techniques to refine the segmentation mask through expansion, tapering, and blurring, ultimately extracting and modifying the face segment within the image for further analysis or visualization.
## Input types
### Required
- **`analysis_models`**
    - A collection of models used for analyzing the image, crucial for determining the initial segmentation mask before refinement.
    - Comfy dtype: `ANALYSIS_MODELS`
    - Python dtype: `object`
- **`image`**
    - The input image to be processed, serving as the basis for face segmentation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`area`**
    - Specifies the area of interest within the image for segmentation, guiding the initial mask creation. This parameter allows selection from predefined areas such as the face, eyes, or mouth, influencing the focus of the segmentation process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list[str]`
- **`grow`**
    - Determines how much to expand the segmentation mask, allowing for more inclusive face coverage.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`grow_tapered`**
    - Indicates whether the mask expansion should taper off, providing a more natural transition at the edges.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`blur`**
    - The intensity of blur applied to the segmentation mask, enhancing the mask's smoothness and blending.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The refined segmentation mask after processing.
    - Python dtype: `torch.Tensor`
- **`image`**
    - Comfy dtype: `IMAGE`
    - The original image modified by the segmentation mask, highlighting the segmented area.
    - Python dtype: `torch.Tensor`
- **`seg_mask`**
    - Comfy dtype: `MASK`
    - A mask representing the precise segment of the face extracted from the image.
    - Python dtype: `torch.Tensor`
- **`seg_image`**
    - Comfy dtype: `IMAGE`
    - The extracted segment of the face from the original image.
    - Python dtype: `torch.Tensor`
- **`x`**
    - Comfy dtype: `INT`
    - The minimum x-coordinate of the face segment within the image.
    - Python dtype: `int`
- **`y`**
    - Comfy dtype: `INT`
    - The minimum y-coordinate of the face segment within the image.
    - Python dtype: `int`
- **`width`**
    - Comfy dtype: `INT`
    - The width of the face segment extracted from the image.
    - Python dtype: `int`
- **`height`**
    - Comfy dtype: `INT`
    - The height of the face segment extracted from the image.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class faceSegmentation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS", ),
                "image": ("IMAGE", ),
                "area": (["face", "main_features", "eyes", "left_eye", "right_eye", "nose", "mouth", "face+forehead (if available)"], ),
                "grow": ("INT", { "default": 0, "min": -4096, "max": 4096, "step": 1 }),
                "grow_tapered": ("BOOLEAN", { "default": False }),
                "blur": ("INT", { "default": 13, "min": 1, "max": 4096, "step": 2 }),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "MASK", "IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("mask", "image", "seg_mask", "seg_image", "x", "y", "width", "height")
    FUNCTION = "segment"
    CATEGORY = "FaceAnalysis"

    def segment(self, analysis_models, image, area, grow, grow_tapered, blur):
        face = tensor_to_image(image[0])

        if face is None:
            raise Exception('No face detected in image')

        landmarks = analysis_models.get_landmarks(face, extended_landmarks=("forehead" in area))

        if area == "face":
            landmarks = landmarks[-2]
        elif area == "eyes":
            landmarks = landmarks[2]
        elif area == "left_eye":
            landmarks = landmarks[3]
        elif area == "right_eye":
            landmarks = landmarks[4]
        elif area == "nose":
            landmarks = landmarks[5]
        elif area == "mouth":
            landmarks = landmarks[6]
        elif area == "main_features":
            landmarks = landmarks[1]
        elif "forehead" in area:
            landmarks = landmarks[-1]

        #mask = np.zeros(face.shape[:2], dtype=np.float64)
        #points = cv2.convexHull(landmarks)
        #cv2.fillConvexPoly(mask, points, color=1)

        mask = mask_from_landmarks(face, landmarks)
        mask = image_to_tensor(mask).unsqueeze(0).squeeze(-1).clamp(0, 1)

        _, y, x = torch.where(mask)
        x1, x2 = x.min().item(), x.max().item()
        y1, y2 = y.min().item(), y.max().item()
        smooth = int(min(max((x2 - x1), (y2 - y1)) * 0.2, 99))

        if smooth > 1:
            if smooth % 2 == 0:
                smooth+= 1
            mask = T.functional.gaussian_blur(mask.bool().unsqueeze(1), smooth).squeeze(1).float()
        
        if grow != 0:
            mask = expand_mask(mask, grow, grow_tapered)

        if blur > 1:
            if blur % 2 == 0:
                blur+= 1
            mask = T.functional.gaussian_blur(mask.unsqueeze(1), blur).squeeze(1).float()

        # extract segment from image
        _, y, x = torch.where(mask)
        x1, x2 = x.min().item(), x.max().item()
        y1, y2 = y.min().item(), y.max().item()
        segment_mask = mask[:, y1:y2, x1:x2]
        segment_image = image[0][y1:y2, x1:x2, :].unsqueeze(0)

        image = image * mask.unsqueeze(-1).repeat(1, 1, 1, 3)

        return (mask, image, segment_mask, segment_image, x1, y1, x2 - x1, y2 - y1,)

```
