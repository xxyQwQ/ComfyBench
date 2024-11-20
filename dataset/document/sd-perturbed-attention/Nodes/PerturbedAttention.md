# Perturbed-Attention Guidance (Advanced)
## Documentation
- Class name: `PerturbedAttention`
- Category: `advanced/model`
- Output node: `False`

The PerturbedAttention node introduces a mechanism to modify the behavior of attention mechanisms within neural network models, specifically targeting the self-attention components of U-Net architectures. It achieves this by applying perturbations to the attention calculations, potentially enhancing the model's ability to focus on relevant features or introducing variability for generative tasks.
## Input types
### Required
- **`model`**
    - The neural network model to which perturbed attention will be applied. This parameter is crucial as it determines the base architecture that will be modified by the perturbation process.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`scale`**
    - Specifies the scale of perturbation to be applied, directly influencing the intensity and impact of the perturbations on the attention mechanism.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`adaptive_scale`**
    - Determines the adaptive scaling factor for perturbations, allowing for dynamic adjustment based on model performance or other criteria.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`unet_block`**
    - Identifies the specific U-Net block (input, middle, or output) where the perturbed attention mechanism will be applied, enabling precise targeting within the model.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`unet_block_id`**
    - The numerical identifier for the U-Net block, further specifying the exact location for perturbation application within the chosen block.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sigma_start`**
    - Defines the starting value of sigma for the perturbation process, controlling the intensity of perturbations applied to the attention mechanism. It plays a key role in adjusting the model's focus and variability.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sigma_end`**
    - Specifies the ending value of sigma, marking the point at which perturbation intensity ceases to increase, thereby defining the range of perturbation application.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`rescale`**
    - A factor that adjusts the scale of perturbations, offering an additional layer of control over their intensity.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`rescale_mode`**
    - Determines how the rescaling of perturbations is applied, with options including 'full' and 'partial', to influence the overall effect on the attention mechanism.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`unet_block_list`**
    - A list specifying the U-Net blocks where the perturbed attention mechanism will be applied. This allows for targeted modifications within the model, enhancing flexibility and control over the perturbation effects.
    - Comfy dtype: `STRING`
    - Python dtype: `List[str]`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified neural network model with perturbed attention mechanisms applied to specified U-Net blocks. This output reflects the enhanced or altered capabilities of the model due to the perturbations.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class PerturbedAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "adaptive_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.0001}),
                "unet_block": (["input", "middle", "output"], {"default": "middle"}),
                "unet_block_id": ("INT", {"default": 0}),
                "sigma_start": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
                "sigma_end": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
                "rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rescale_mode": (["full", "partial"], {"default": "full"}),
            },
            "optional": {
                "unet_block_list": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def parse_unet_blocks(self, model: ModelPatcher, unet_block_list: str):
        output: list[tuple[str, int, int | None]] = []

        # Get all Self-attention blocks
        input_blocks, middle_blocks, output_blocks = [], [], []
        for name, module in model.model.diffusion_model.named_modules():
            if module.__class__.__name__ == "CrossAttention" and name.endswith("attn1"):
                parts = name.split(".")
                block_name = parts[0]
                block_id = int(parts[1])
                if block_name.startswith("input"):
                    input_blocks.append(block_id)
                elif block_name.startswith("middle"):
                    middle_blocks.append(block_id - 1)
                elif block_name.startswith("output"):
                    output_blocks.append(block_id)

        def group_blocks(blocks: list[int]):
            return [(i, len(list(gr))) for i, gr in groupby(blocks)]

        input_blocks, middle_blocks, output_blocks = group_blocks(input_blocks), group_blocks(middle_blocks), group_blocks(output_blocks)

        unet_blocks = [b.strip() for b in unet_block_list.split(",")]
        for block in unet_blocks:
            name, indices = block[0], block[1:].split(".")
            match name:
                case "d":
                    layer, cur_blocks = "input", input_blocks
                case "m":
                    layer, cur_blocks = "middle", middle_blocks
                case "u":
                    layer, cur_blocks = "output", output_blocks
            if len(indices) >= 2:
                number, index = cur_blocks[int(indices[0])][0], int(indices[1])
                assert 0 <= index < cur_blocks[int(indices[0])][1]
            else:
                number, index = cur_blocks[int(indices[0])][0], None
            output.append((layer, number, index))

        return output

    def patch(
        self,
        model: ModelPatcher,
        scale: float = 3.0,
        adaptive_scale: float = 0.0,
        unet_block: str = "middle",
        unet_block_id: int = 0,
        sigma_start: float = -1.0,
        sigma_end: float = -1.0,
        rescale: float = 0.0,
        rescale_mode: str = "full",
        unet_block_list: str = "",
    ):
        m = model.clone()

        sigma_start = float("inf") if sigma_start < 0 else sigma_start
        if unet_block_list:
            blocks = self.parse_unet_blocks(model, unet_block_list)
        else:
            blocks = [(unet_block, unet_block_id, None)]

        def perturbed_attention(q: Tensor, k: Tensor, v: Tensor, extra_options, mask=None):
            """Perturbed self-attention"""
            return v

        def post_cfg_function(args):
            """CFG+PAG"""
            model = args["model"]
            cond_pred = args["cond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            signal_scale = scale
            if adaptive_scale > 0:
                t = model.model_sampling.timestep(sigma)[0].item()
                signal_scale -= scale * (adaptive_scale**4) * (1000 - t)
                if signal_scale < 0:
                    signal_scale = 0

            if signal_scale == 0 or not (sigma_end < sigma[0] <= sigma_start):
                return cfg_result

            # Replace Self-attention with PAG
            for block in blocks:
                layer, number, index = block
                model_options = set_model_options_patch_replace(model_options, perturbed_attention, "attn1", layer, number, index)

            if BACKEND == "ComfyUI":
                (pag_cond_pred,) = calc_cond_batch(model, [cond], x, sigma, model_options)
            if BACKEND == "Forge":
                (pag_cond_pred, _) = calc_cond_uncond_batch(model, cond, None, x, sigma, model_options)

            pag = (cond_pred - pag_cond_pred) * signal_scale

            return cfg_result + rescale_pag(pag, cond_pred, cfg_result, rescale, rescale_mode)

        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return (m,)

```
