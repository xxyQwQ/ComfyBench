---
tags:
- BoundingBox
- Image
- ImageTransformation
---

# Face Bounding Box
## Documentation
- Class name: `FaceBoundingBox`
- Category: `FaceAnalysis`
- Output node: `False`

The FaceBoundingBox node is designed to detect and extract bounding boxes around faces in images. It utilizes analysis models to identify the location and dimensions of faces, applying optional padding and selecting specific faces based on an index. This functionality is crucial for tasks that require precise face detection and cropping, such as facial recognition or analysis applications.
## Input types
### Required
- **`analysis_models`**
    - Specifies the models used for face detection and analysis. It's crucial for determining the accuracy and efficiency of face detection within images.
    - Comfy dtype: `ANALYSIS_MODELS`
    - Python dtype: `object`
- **`image`**
    - The input image or images in which faces need to be detected. It serves as the primary data for face detection.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`padding`**
    - An optional parameter that adds a fixed padding to the detected face bounding boxes, allowing for more context around the faces.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`padding_percent`**
    - Similar to padding, but adds a percentage-based padding around the detected face bounding boxes, offering flexible context inclusion.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`index`**
    - Specifies the index of the face to focus on when multiple faces are detected. A negative index indicates no specific focus, utilizing all detected faces.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - The cropped images of detected faces, potentially adjusted by padding and index selection.
    - Python dtype: `List[torch.Tensor]`
- **`x`**
    - Comfy dtype: `INT`
    - The x-coordinate of the top-left corner of each face bounding box.
    - Python dtype: `List[int]`
- **`y`**
    - Comfy dtype: `INT`
    - The y-coordinate of the top-left corner of each face bounding box.
    - Python dtype: `List[int]`
- **`width`**
    - Comfy dtype: `INT`
    - The width of each face bounding box.
    - Python dtype: `List[int]`
- **`height`**
    - Comfy dtype: `INT`
    - The height of each face bounding box.
    - Python dtype: `List[int]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class FaceBoundingBox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS", ),
                "image": ("IMAGE", ),
                "padding": ("INT", { "default": 0, "min": 0, "max": 4096, "step": 1 }),
                "padding_percent": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05 }),
                "index": ("INT", { "default": -1, "min": -1, "max": 4096, "step": 1 }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "x", "y", "width", "height")
    FUNCTION = "bbox"
    CATEGORY = "FaceAnalysis"
    OUTPUT_IS_LIST = (True, True, True, True, True,)

    def bbox(self, analysis_models, image, padding, padding_percent, index=-1):
        out_img = []
        out_x = []
        out_y = []
        out_w = []
        out_h = []

        for i in image:
            i = T.ToPILImage()(i.permute(2, 0, 1)).convert('RGB')
            img, x, y, w, h = analysis_models.get_bbox(i, padding, padding_percent)
            out_img.extend(img)
            out_x.extend(x)
            out_y.extend(y)
            out_w.extend(w)
            out_h.extend(h)

        if not out_img:
            raise Exception('No face detected in image.')

        if len(out_img) == 1:
            index = 0

        if index > len(out_img) - 1:
            index = len(out_img) - 1

        if index != -1:
            out_img = [out_img[index]]
            out_x = [out_x[index]]
            out_y = [out_y[index]]
            out_w = [out_w[index]]
            out_h = [out_h[index]]
        #else:
        #    w = out_img[0].shape[1]
        #    h = out_img[0].shape[0]

            #out_img = [comfy.utils.common_upscale(img.unsqueeze(0).movedim(-1,1), w, h, "bilinear", "center").movedim(1,-1).squeeze(0) for img in out_img]
            #out_img = torch.stack(out_img)
        
        return (out_img, out_x, out_y, out_w, out_h,)

```
