---
tags:
- Mask
---

# Mask Or Image To Weight
## Documentation
- Class name: `MaskOrImageToWeight`
- Category: `KJNodes/weights`
- Output node: `False`

The MaskOrImageToWeight node is designed to calculate the mean value of either masks or images provided as input, but not both simultaneously. It supports converting these mean values into different output types, including lists, pandas series, or tensors, based on the specified output type. This functionality allows for flexible data handling and integration into broader data processing or machine learning workflows.
## Input types
### Required
- **`output_type`**
    - Specifies the desired format for the output data, which can be a list, pandas series, or tensor. This choice determines how the mean values calculated from the input masks or images are structured for subsequent use.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`images`**
    - An optional input consisting of images from which mean values are calculated if masks are not provided. This parameter allows the node to work with either image or mask data, but not both simultaneously.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Optional[List[torch.Tensor]]`
- **`masks`**
    - An optional input consisting of masks from which mean values are calculated if images are not provided. This parameter enables the node to process either mask or image inputs, ensuring flexibility in data handling.
    - Comfy dtype: `MASK`
    - Python dtype: `Optional[List[torch.Tensor]]`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - The processed mean values of the input masks or images, formatted according to the specified output type. This output facilitates further data manipulation or analysis in subsequent steps of a pipeline.
    - Python dtype: `Union[List[float], pandas.Series, torch.Tensor]`
- **`string`**
    - Comfy dtype: `STRING`
    - A list of the mean values calculated from the input, represented as strings. This provides an alternative view of the data for logging or display purposes.
    - Python dtype: `List[str]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MaskOrImageToWeight:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "output_type": (
                [   
                    'list',
                    'pandas series',
                    'tensor',
                    'string'
                ],
                {
                "default": 'list'
                    }),
             },
            "optional": {
                "images": ("IMAGE",),
                "masks": ("MASK",),                
            },

        }
    RETURN_TYPES = ("FLOAT", "STRING",)
    FUNCTION = "execute"
    CATEGORY = "KJNodes/weights"
    DESCRIPTION = """
Gets the mean values from mask or image batch  
and returns that as the selected output type.   
"""

    def execute(self, output_type, images=None, masks=None):
        mean_values = []
        if masks is not None and images is None:
            for mask in masks:
                mean_values.append(mask.mean().item())
        elif masks is None and images is not None:
            for image in images:
                mean_values.append(image.mean().item())
        elif masks is not None and images is not None:
            raise Exception("MaskOrImageToWeight: Use either mask or image input only.")
                  
        # Convert mean_values to the specified output_type
        if output_type == 'list':
            out = mean_values,
        elif output_type == 'pandas series':
            try:
                import pandas as pd
            except:
                raise Exception("MaskOrImageToWeight: pandas is not installed. Please install pandas to use this output_type")
            out = pd.Series(mean_values),
        elif output_type == 'tensor':
            out = torch.tensor(mean_values, dtype=torch.float32),
        return (out, [str(value) for value in mean_values],)

```
