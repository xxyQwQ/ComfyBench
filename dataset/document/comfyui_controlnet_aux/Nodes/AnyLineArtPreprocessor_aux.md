---
tags:
- DepthMap
- Image
- ImagePreprocessing
- LineExtraction
---

# AnyLine Lineart
## Documentation
- Class name: `AnyLineArtPreprocessor_aux`
- Category: `ControlNet Preprocessors/Line Extractors`
- Output node: `False`

This node is designed to preprocess images by applying a line art extraction process, which can be customized to merge with different line art styles such as standard, realistic, anime, or manga. It enhances images for further processing or visualization by extracting prominent lines and adjusting them according to the selected line art style, making it versatile for various artistic and design applications.
## Input types
### Required
- **`image`**
    - The input image to be processed for line art extraction. It serves as the primary data on which the line art preprocessing is performed, determining the visual output of the node.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`merge_with_lineart`**
    - Specifies the line art style to merge with during the preprocessing, such as 'lineart_standard', 'lineart_realistic', 'lineart_anime', or 'manga_line'. This choice influences the aesthetic and stylistic outcome of the processed image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`resolution`**
    - Defines the resolution for the output image, affecting the detail and quality of the extracted line art.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`lineart_lower_bound`**
    - Sets the lower bound for line art extraction, influencing the minimum intensity of lines to be included in the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`lineart_upper_bound`**
    - Sets the upper bound for line art extraction, influencing the maximum intensity of lines to be included in the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`object_min_size`**
    - Determines the minimum size of objects to be considered in the line art extraction, affecting the granularity of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`object_connectivity`**
    - Specifies the connectivity criteria for objects in the line art extraction, impacting how individual lines and shapes are interpreted and connected in the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The processed image with extracted line art, presented in a format suitable for visualization or further artistic processing.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class AnyLinePreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "merge_with_lineart": (["lineart_standard", "lineart_realisitic", "lineart_anime", "manga_line"], {"default": "lineart_standard"}),
                "resolution": ("INT", {"default": 1280, "min": 512, "max": MAX_RESOLUTION, "step": 8})
            },
            "optional": {
                "lineart_lower_bound": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
                "lineart_upper_bound": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                "object_min_size": ("INT", {"default": 36, "min": 1, "max": MAX_RESOLUTION}),
                "object_connectivity": ("INT", {"default": 1, "min": 1, "max": MAX_RESOLUTION}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "get_anyline"
    CATEGORY = "ControlNet Preprocessors/Line Extractors"

    def __init__(self):
        self.device = model_management.get_torch_device()

    def get_anyline(self, image, merge_with_lineart, resolution, lineart_lower_bound=0, lineart_upper_bound=1, object_min_size=36, object_connectivity=1):
        from controlnet_aux.teed import TEDDetector
        from skimage import morphology
        pbar = comfy.utils.ProgressBar(3)

        # Process the image with MTEED model
        mteed_model = TEDDetector.from_pretrained("TheMistoAI/MistoLine", "MTEED.pth", subfolder="Anyline").to(self.device)
        mteed_result = common_annotator_call(mteed_model, image, resolution=resolution, show_pbar=False)
        mteed_result = mteed_result.numpy()
        del mteed_model
        pbar.update(1)

        # Process the image with the lineart standard preprocessor
        if merge_with_lineart == "lineart_standard":
            from controlnet_aux.lineart_standard import LineartStandardDetector
            lineart_standard_detector = LineartStandardDetector()
            lineart_result = common_annotator_call(lineart_standard_detector, image, guassian_sigma=2, intensity_threshold=3, resolution=resolution, show_pbar=False).numpy()
            del lineart_standard_detector
        else:
            from controlnet_aux.lineart import LineartDetector
            from controlnet_aux.lineart_anime import LineartAnimeDetector
            from controlnet_aux.manga_line import LineartMangaDetector
            lineart_detector = dict(lineart_realisitic=LineartDetector, lineart_anime=LineartAnimeDetector, manga_line=LineartMangaDetector)[merge_with_lineart]
            lineart_detector = lineart_detector.from_pretrained().to(self.device)
            lineart_result = common_annotator_call(lineart_detector, image, resolution=resolution, show_pbar=False).numpy()
            del lineart_detector
        pbar.update(1)
        
        final_result = []
        for i in range(len(image)):
            _lineart_result  = get_intensity_mask(lineart_result[i], lower_bound=lineart_lower_bound, upper_bound=lineart_upper_bound)
            _cleaned = morphology.remove_small_objects(_lineart_result.astype(bool), min_size=object_min_size, connectivity=object_connectivity)
            _lineart_result = _lineart_result * _cleaned
            _mteed_result = mteed_result[i]

            # Combine the results
            final_result.append(torch.from_numpy(combine_layers(_mteed_result, _lineart_result)))
        pbar.update(1)
        return (torch.stack(final_result),)

```
