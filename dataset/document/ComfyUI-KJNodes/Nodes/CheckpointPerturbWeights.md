# CheckpointPerturbWeights
## Documentation
- Class name: `CheckpointPerturbWeights`
- Category: `KJNodes/experimental`
- Output node: `True`

This node is designed to modify the weights of a model checkpoint in a controlled manner, allowing for the introduction of perturbations or adjustments to the model's parameters. Its purpose is to facilitate experiments with model behavior under slightly altered conditions, potentially improving performance or robustness through fine-tuning.
## Input types
### Required
- **`model`**
    - Specifies the model whose weights are to be perturbed, serving as the foundation for the perturbation process.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`joint_blocks`**
    - Determines the magnitude of perturbation applied to the weights of joint blocks within the model, influencing the extent of modification.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`final_layer`**
    - Controls the perturbation level for the model's final layer weights, directly affecting the output characteristics.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`rest_of_the_blocks`**
    - Sets the perturbation intensity for the weights of all other blocks in the model, excluding the final and joint blocks.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`seed`**
    - Provides a seed value for random number generation, ensuring reproducibility of the perturbation effects.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - Returns the model with its weights perturbed according to the specified parameters, ready for further use or analysis.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CheckpointPerturbWeights:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "joint_blocks": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 10.0, "step": 0.001}),
            "final_layer": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 10.0, "step": 0.001}),
            "rest_of_the_blocks": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 10.0, "step": 0.001}),
            "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "mod"
    OUTPUT_NODE = True

    CATEGORY = "KJNodes/experimental"

    def mod(self, seed, model, joint_blocks, final_layer, rest_of_the_blocks):
        import copy
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        device = model_management.get_torch_device()
        model_copy = copy.deepcopy(model)
        model_copy.model.to(device)
        keys = model_copy.model.diffusion_model.state_dict().keys()

        dict = {}
        for key in keys:
            dict[key] = model_copy.model.diffusion_model.state_dict()[key]

        pbar = ProgressBar(len(keys))
        for k in keys:
            v = dict[k]
            print(f'{k}: {v.std()}') 
            if k.startswith('joint_blocks'):
                multiplier = joint_blocks
            elif k.startswith('final_layer'):
                multiplier = final_layer
            else:
                multiplier = rest_of_the_blocks
            dict[k] += torch.normal(torch.zeros_like(v) * v.mean(), torch.ones_like(v) * v.std() * multiplier).to(device)
            pbar.update(1)
        model_copy.model.diffusion_model.load_state_dict(dict)
        return model_copy,

```
