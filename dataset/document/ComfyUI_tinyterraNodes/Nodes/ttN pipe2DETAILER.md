---
tags:
- Image
- Pipeline
- PipelineTransformation
---

# pipe > detailer_pipe
## Documentation
- Class name: `ttN pipe2DETAILER`
- Category: `🌏 tinyterra/pipe`
- Output node: `False`

The ttN pipe2DETAILER node is designed to enhance and detail the data pipeline by integrating additional processing steps. It focuses on refining and enriching the pipeline's output, making it more detailed and tailored for subsequent operations or analyses.
## Input types
### Required
- **`pipe`**
    - The 'pipe' input is the primary data structure that carries the flow of data through the node. It is essential for the node's operation, serving as the basis for the detailing process.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `dict`
- **`bbox_detector`**
    - The 'bbox_detector' input specifies the bounding box detector to be used for identifying regions of interest within the data. This is crucial for the detailing process, as it allows for targeted enhancements.
    - Comfy dtype: `BBOX_DETECTOR`
    - Python dtype: `object`
- **`wildcard`**
    - The 'wildcard' input allows for flexible specification of additional conditions or parameters that can be applied during the detailing process. It enables customization and fine-tuning of the output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`sam_model_opt`**
    - An optional input for specifying a SAM model to be used in the detailing process. It provides an opportunity to apply specialized attention mechanisms for improved detail and accuracy.
    - Comfy dtype: `SAM_MODEL`
    - Python dtype: `object`
- **`segm_detector_opt`**
    - An optional input for specifying a segmentation detector. This enhances the detailing process by allowing for precise segmentation of areas within the data, contributing to more focused enhancements.
    - Comfy dtype: `SEGM_DETECTOR`
    - Python dtype: `object`
- **`detailer_hook`**
    - An optional input that allows for the integration of custom detailing functions or hooks. This enables the node to perform highly customized detailing operations based on specific requirements.
    - Comfy dtype: `DETAILER_HOOK`
    - Python dtype: `object`
## Output types
- **`detailer_pipe`**
    - Comfy dtype: `DETAILER_PIPE`
    - The 'detailer_pipe' output represents the enhanced and detailed version of the input pipeline, enriched with additional data and processing steps.
    - Python dtype: `tuple`
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - The 'pipe' output provides the original input pipeline, allowing for comparison or further processing alongside the detailed version.
    - Python dtype: `dict`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ttN_pipe_2DETAILER:
    version = '1.2.0'
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"pipe": ("PIPE_LINE",),
                             "bbox_detector": ("BBOX_DETECTOR", ), 
                             "wildcard": ("STRING", {"multiline": True, "placeholder": "wildcard spec: if kept empty, this option will be ignored"}),
                            },
                "optional": {"sam_model_opt": ("SAM_MODEL", ), 
                             "segm_detector_opt": ("SEGM_DETECTOR",),
                             "detailer_hook": ("DETAILER_HOOK",),
                            },
                "hidden": {"ttNnodeVersion": ttN_pipe_2DETAILER.version},
                }

    RETURN_TYPES = ("DETAILER_PIPE", "PIPE_LINE" )
    RETURN_NAMES = ("detailer_pipe", "pipe")
    FUNCTION = "flush"

    CATEGORY = "🌏 tinyterra/pipe"

    def flush(self, pipe, bbox_detector, wildcard, sam_model_opt=None, segm_detector_opt=None, detailer_hook=None):
        detailer_pipe = (pipe.get('model'), pipe.get('clip'), pipe.get('vae'), pipe.get('positive'), pipe.get('negative'), wildcard,
                         bbox_detector, segm_detector_opt, sam_model_opt, detailer_hook, None, None, None, None)
        return (detailer_pipe, pipe, )

```
