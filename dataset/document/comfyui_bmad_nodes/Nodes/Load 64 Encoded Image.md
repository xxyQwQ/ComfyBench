---
tags:
- Image
---

# Load 64 Encoded Image
## Documentation
- Class name: `Load 64 Encoded Image`
- Category: `Bmad/api`
- Output node: `False`

This node is designed to decode and process a base64-encoded image, converting it into a tensor representation suitable for further image processing tasks. It emphasizes the transformation of encoded image data into a standardized format that can be seamlessly integrated into image analysis and manipulation workflows.
## Input types
### Required
- **`image_code`**
    - The base64-encoded string of the image to be loaded. This parameter is crucial for decoding the image and converting it into a tensor format for subsequent operations.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The decoded and processed image, represented as a tensor. This output is ready for use in various image processing and analysis tasks.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class LoadImage64:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "image_code": ("STRING", {"default": "insert encoded image here"})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_image"
    CATEGORY = api_category_path

    def get_image(self, image_code):
        image = Image.open(BytesIO(base64.b64decode(image_code)))
        image = image.convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image,)

```
