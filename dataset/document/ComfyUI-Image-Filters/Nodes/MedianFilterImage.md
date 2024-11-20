# Median Filter Image
## Documentation
- Class name: `MedianFilterImage`
- Category: `image/filters`
- Output node: `False`

The MedianFilterImage node applies a median filter to a batch of images, smoothing them by replacing each pixel's value with the median value of the intensities in its neighborhood. This process is particularly effective at reducing salt-and-pepper noise while preserving edges, making it a valuable tool for image preprocessing and noise reduction.
## Input types
### Required
- **`images`**
    - The 'images' parameter represents the batch of images to be processed. It is crucial for defining the input data on which the median filtering operation will be performed, directly influencing the output's quality and effectiveness.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`size`**
    - The 'size' parameter determines the size of the neighborhood used for calculating the median value for each pixel. It affects the degree of smoothing and noise reduction, with larger sizes leading to more pronounced filtering effects.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a batch of images that have been smoothed using the median filter. This process reduces noise while preserving the essential structure and edges of the images.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MedianFilterImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "size": ("INT", {"default": 1, "min": 1, "max": 1023}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "filter_image"

    CATEGORY = "image/filters"

    def filter_image(self, images, size):
        np_images = images.detach().clone().cpu().numpy()
        d = size * 2 + 1
        for index, image in enumerate(np_images):
            if d > 5:
                work_image = image * 255
                work_image = cv2.medianBlur(work_image.astype(np.uint8), d)
                np_images[index] = work_image.astype(np.float32) / 255
            else:
                np_images[index] = cv2.medianBlur(image, d)
        return (torch.from_numpy(np_images),)

```
