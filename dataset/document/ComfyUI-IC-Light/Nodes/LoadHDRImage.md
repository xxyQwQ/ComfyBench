---
tags:
- Image
---

# Load HDR Image
## Documentation
- Class name: `LoadHDRImage`
- Category: `IC-Light`
- Output node: `False`

The LoadHDRImage node is designed to load High Dynamic Range (HDR) images from a specified directory, process them to simulate different exposures, and convert them into a batch of Low Dynamic Range (LDR) images. This node facilitates the exploration of various exposure levels within an HDR image by generating multiple exposure-adjusted versions, thereby enabling enhanced image analysis and manipulation.
## Input types
### Required
- **`image`**
    - Specifies the HDR image to be loaded and processed. The image is identified by its name within the input directory, and this parameter is crucial for determining which HDR image will undergo exposure adjustments.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`exposures`**
    - A comma-separated string representing the exposure levels to apply to the HDR image. These values are used to simulate different lighting conditions by adjusting the brightness of the image, thereby generating multiple LDR versions.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - Represents the LDR images generated from the HDR image at specified exposure levels, facilitating further image processing or analysis.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - This output type is not directly mentioned in the provided context, indicating a potential misunderstanding or error in the original feedback.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class LoadHDRImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": False}),
                     "exposures": ("STRING", {"default": "-2,-1,0,1,2"}),
                     },
                }

    CATEGORY = "IC-Light"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "loadhdrimage"
    DESCRIPTION = """
Loads a .hdr image from the input directory.  
Output is a batch of LDR images with the selected exposures.  

"""
    def loadhdrimage(self, image, exposures):
        import cv2
        image_path = folder_paths.get_annotated_filepath(image)
        # Load the HDR image
        hdr_image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)

        exposures = list(map(int, exposures.split(",")))
        if not isinstance(exposures, list):
            exposures = [exposures]  # Example exposure values
        ldr_images_tensors = []

        for exposure in exposures:
            # Scale pixel values to simulate different exposures
            ldr_image = np.clip(hdr_image * (2**exposure), 0, 1)
            # Convert to 8-bit image (LDR) by scaling to 255
            ldr_image_8bit = np.uint8(ldr_image * 255)
            # Convert BGR to RGB
            ldr_image_8bit = cv2.cvtColor(ldr_image_8bit, cv2.COLOR_BGR2RGB)
            # Convert the LDR image to a torch tensor
            tensor_image = torch.from_numpy(ldr_image_8bit).float()
            # Normalize the tensor to the range [0, 1]
            tensor_image = tensor_image / 255.0
            # Change the tensor shape to (C, H, W)
            tensor_image = tensor_image.permute(2, 0, 1)
            # Add the tensor to the list
            ldr_images_tensors.append(tensor_image)

        batch_tensors = torch.stack(ldr_images_tensors)
        batch_tensors = batch_tensors.permute(0, 2, 3, 1)

        return batch_tensors,

```
