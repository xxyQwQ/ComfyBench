---
tags:
- Animation
- PoseEstimation
---

# Pose Node
## Documentation
- Class name: `PoseNode`
- Category: `AlekPet Nodes/image`
- Output node: `False`

The PoseNode is designed to process images by converting them into a tensor format suitable for pose estimation tasks. It emphasizes the preparation of image data for further analysis or model inference, focusing on the transformation of images from their original format into a normalized tensor representation.
## Input types
### Required
- **`image`**
    - The 'image' parameter represents the name of the image file to be processed. It is crucial for locating and loading the image from a temporary directory, where it undergoes conversion and normalization for pose analysis.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a tensor representation of the input image, prepared for pose estimation tasks. This tensor is normalized and ready for further processing or model inference.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class PoseNode(object):
    @classmethod
    def INPUT_TYPES(self):
        temp_dir = folder_paths.get_temp_directory()

        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)

        temp_dir = folder_paths.get_temp_directory()

        return {
            "required": {"image": (sorted(os.listdir(temp_dir)),)},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "output_pose"

    CATEGORY = "AlekPet Nodes/image"

    def output_pose(self, image):
        image_path = os.path.join(folder_paths.get_temp_directory(), image)
        # print(f"Create: {image_path}")

        i = Image.open(image_path)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image,)

    @classmethod
    def IS_CHANGED(self, image):
        image_path = os.path.join(folder_paths.get_temp_directory(), image)
        # print(f'Change: {image_path}')

        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

```
