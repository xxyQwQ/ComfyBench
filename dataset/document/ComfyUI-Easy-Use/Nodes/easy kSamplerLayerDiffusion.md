---
tags:
- LayeredDiffusion
---

# EasyKSampler (LayerDiffuse)
## Documentation
- Class name: `easy kSamplerLayerDiffusion`
- Category: `EasyUse/Sampler`
- Output node: `True`

This node integrates the concept of layer diffusion with the kSampler process in a simplified manner, aiming to enhance image generation by applying specialized diffusion techniques. It focuses on adjusting the blending and diffusion parameters to optimize the visual output based on the layer diffusion method selected, thereby facilitating a more controlled and refined image synthesis process.
## Input types
### Required
- **`pipe`**
    - Represents the pipeline configuration, serving as the foundational structure for the layer diffusion process. It is crucial for defining the flow and parameters of the image generation task. The inclusion of this parameter directly influences the customization and execution of the diffusion and blending techniques applied to the image generation.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `dict`
- **`image_output`**
    - Determines the format in which the generated images are outputted, affecting the visualization and saving options. This parameter allows for flexibility in how the results are presented and stored, catering to different preferences and requirements.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`link_id`**
    - Specifies a unique identifier for the link within the pipeline, enabling the tracking and association of generated images with specific pipeline executions. This parameter plays a key role in managing and organizing the output images in complex workflows.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`save_prefix`**
    - Defines a prefix for saving generated images, facilitating organized storage and retrieval of output files. This parameter is essential for categorizing and identifying images in large-scale image generation tasks.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`model`**
    - The model parameter allows for the selection of a specific generative model to be used in the layer diffusion process. This choice significantly impacts the quality and characteristics of the generated images, enabling fine-tuned control over the synthesis outcome.
    - Comfy dtype: `MODEL`
    - Python dtype: `object`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - Outputs the modified pipeline configuration, incorporating the adjustments made during the layer diffusion process. This reflects the changes and optimizations applied to the diffusion and blending settings.
    - Python dtype: `dict`
- **`final_image`**
    - Comfy dtype: `IMAGE`
    - The final generated image after the layer diffusion process, showcasing the enhanced visual quality and detail achieved through the applied techniques.
    - Python dtype: `object`
- **`original_image`**
    - Comfy dtype: `IMAGE`
    - Provides the original image before the application of layer diffusion, allowing for comparison and evaluation of the diffusion effect on the image quality.
    - Python dtype: `object`
- **`alpha`**
    - Comfy dtype: `MASK`
    - Outputs the alpha channel information of the generated images, which is crucial for understanding and manipulating transparency and blending effects in post-processing stages.
    - Python dtype: `list`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class samplerSimpleLayerDiffusion(samplerFull):

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save"],{"default": "Preview"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"})
                 },
                "optional": {
                    "model": ("MODEL",),
                },
                "hidden": {
                    "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("pipe", "final_image", "original_image", "alpha")
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False, False, False, True)
    FUNCTION = "layerDiffusion"
    CATEGORY = "EasyUse/Sampler"

    def layerDiffusion(self, pipe, image_output='preview', link_id=0, save_prefix='ComfyUI', model=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):

        result = super().run(pipe, None, None,None,None,None, image_output, link_id, save_prefix,
                               None, model, None, None, None, None, None, None,
                               None, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise)
        pipe = result["result"][0] if "result" in result else None
        return ({"ui":result['ui'], "result":(pipe, pipe["images"], pipe["samp_images"], pipe["alpha"])})

```
