---
tags:
- IPAdapter
---

# IP Adapter Apply
## Documentation
- Class name: `AV_IPAdapter`
- Category: `Art Venture/IP Adapter`
- Output node: `False`

The AV_IPAdapter node is designed to integrate and apply image processing adapters to models, enhancing their capabilities with additional image processing techniques. It allows for the dynamic application of IP adapters and clip vision models to modify and improve the input model's performance on image data, based on specified weights, noise levels, and optional configurations.
## Input types
### Required
- **`ip_adapter_name`**
    - Specifies the name of the IP adapter to be used. It is crucial for determining which IP adapter model to load and apply.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`clip_name`**
    - Defines the name of the clip vision model to be used alongside the IP adapter. This parameter is essential for selecting the appropriate clip vision model to complement the IP adapter's functionality.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`model`**
    - The model to which the IP adapter and clip vision modifications will be applied. This parameter is central to the node's operation, serving as the base for enhancements.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`image`**
    - The image data to be processed by the IP adapter and clip vision models. This input is key to the adaptation and enhancement process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`weight`**
    - Affects the intensity of the IP adapter's application on the image/model. It allows for fine-tuning the strength of the modifications.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`noise`**
    - Introduces a level of noise to the adaptation process, offering a means to adjust the effect of the IP adapter on the image/model.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`ip_adapter_opt`**
    - Optional configurations for the IP adapter, providing flexibility in customizing its application.
    - Comfy dtype: `IPADAPTER`
    - Python dtype: `Dict`
- **`clip_vision_opt`**
    - Optional configurations for the clip vision model, enabling further customization of the clip vision's application.
    - Comfy dtype: `CLIP_VISION`
    - Python dtype: `Dict`
- **`attn_mask`**
    - unknown
    - Comfy dtype: `MASK`
    - Python dtype: `unknown`
- **`start_at`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`end_at`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`weight_type`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`enabled`**
    - Controls whether the IP adapter and clip vision modifications are applied, allowing for conditional processing.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The enhanced model after applying the IP adapter and clip vision modifications.
    - Python dtype: `torch.nn.Module`
- **`pipeline`**
    - Comfy dtype: `IPADAPTER`
    - A dictionary detailing the applied IP adapter and clip vision models, offering insight into the modifications made.
    - Python dtype: `Dict`
- **`clip_vision`**
    - Comfy dtype: `CLIP_VISION`
    - The specific clip vision model applied during the process, providing context for the enhancements.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [KSampler (Efficient)](../../efficiency-nodes-comfyui/Nodes/KSampler (Efficient).md)



## Source code
```python
    class AV_IPAdapter(IPAdapterModelLoader, IPAdapterApply):
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "ip_adapter_name": (["None"] + folder_paths.get_filename_list("ipadapter"),),
                    "clip_name": (["None"] + folder_paths.get_filename_list("clip_vision"),),
                    "model": ("MODEL",),
                    "image": ("IMAGE",),
                    "weight": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                    "noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                },
                "optional": {
                    "ip_adapter_opt": ("IPADAPTER",),
                    "clip_vision_opt": ("CLIP_VISION",),
                    "attn_mask": ("MASK",),
                    "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "weight_type": (
                        ["standard", "prompt is more important", "style transfer (SDXL only)"],
                        {"default": "standard"},
                    ),
                    "enabled": ("BOOLEAN", {"default": True}),
                },
            }

        RETURN_TYPES = ("MODEL", "IPADAPTER", "CLIP_VISION")
        RETURN_NAMES = ("model", "pipeline", "clip_vision")
        CATEGORY = "Art Venture/IP Adapter"
        FUNCTION = "apply_ip_adapter"

        def apply_ip_adapter(
            self,
            ip_adapter_name,
            clip_name,
            model,
            image,
            weight,
            noise,
            ip_adapter_opt=None,
            clip_vision_opt=None,
            enabled=True,
            **kwargs,
        ):
            if not enabled:
                return (model, None, None)

            if ip_adapter_opt:
                if "ipadapter" in ip_adapter_opt:
                    ip_adapter = ip_adapter_opt["ipadapter"]["model"]
                else:
                    ip_adapter = ip_adapter_opt
            else:
                assert ip_adapter_name != "None", "IP Adapter name must be specified"
                ip_adapter = loader.load_ipadapter_model(ip_adapter_name)[0]

            if clip_vision_opt:
                clip_vision = clip_vision_opt
            elif ip_adapter_opt and "clipvision" in ip_adapter_opt:
                clip_vision = ip_adapter_opt["clipvision"]["model"]
            else:
                assert clip_name != "None", "Clip vision name must be specified"
                clip_path = folder_paths.get_full_path("clip_vision", clip_name)
                clip_vision = comfy.clip_vision.load(clip_path)

            pipeline = {"ipadapter": {"model": ip_adapter}, "clipvision": {"model": clip_vision}}

            res: Tuple = apply.apply_ipadapter(model, pipeline, image=image, weight=weight, **kwargs)
            res += (pipeline, clip_vision)

            return res

```
