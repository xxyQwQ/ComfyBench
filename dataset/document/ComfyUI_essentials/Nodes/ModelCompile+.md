---
tags:
- Model
---

# 🔧 Model Compile
## Documentation
- Class name: `ModelCompile+`
- Category: `essentials/utilities`
- Output node: `False`

The ModelCompile node is designed for optimizing and compiling models for improved execution performance. It allows for dynamic adjustments and fine-tuning of the compilation process based on the model's needs and the desired optimization level, enhancing efficiency and potentially reducing overhead.
## Input types
### Required
- **`model`**
    - The model to be compiled. This parameter is crucial as it determines the base model that will undergo the compilation process for optimization.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`fullgraph`**
    - A boolean flag indicating whether to compile the model using a full graph approach. This affects the depth and extent of the compilation, potentially leading to more thorough optimization.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`dynamic`**
    - A boolean flag that enables dynamic compilation, allowing for more flexibility in how the model is optimized during the compilation process.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`mode`**
    - Specifies the mode of compilation, which can range from default settings to modes focused on reducing overhead or maximizing autotuning for performance optimization.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The compiled and optimized model, ready for enhanced performance during execution.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ModelCompile():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "fullgraph": ("BOOLEAN", { "default": False }),
                "dynamic": ("BOOLEAN", { "default": False }),
                "mode": (["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],),
            },
        }

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "execute"
    CATEGORY = "essentials/utilities"

    def execute(self, model, fullgraph, dynamic, mode):
        work_model = model.clone()
        torch._dynamo.config.suppress_errors = True
        work_model.model.diffusion_model = torch.compile(work_model.model.diffusion_model, dynamic=dynamic, fullgraph=fullgraph, mode=mode)
        return (work_model, )

```
