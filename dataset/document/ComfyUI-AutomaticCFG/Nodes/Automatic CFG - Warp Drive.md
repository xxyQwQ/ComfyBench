# Automatic CFG - Warp Drive
## Documentation
- Class name: `Automatic CFG - Warp Drive`
- Category: `model_patches/Automatic_CFG/presets`
- Output node: `False`

This node engages an advanced configuration for dynamic control flow graph (CFG) modification in a 'Warp Drive' mode, aimed at maximizing the speed and efficiency of model adjustments. It suggests settings for different levels of risk and performance, emphasizing speed and experimental adjustments for clearer subject prompts or general use.
## Input types
### Required
- **`model`**
    - The model to be patched with advanced dynamic CFG settings for enhanced performance and speed.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`uncond_sigma_start`**
    - Specifies the starting value of unconditional sigma for the dynamic CFG adjustments, influencing the model's behavior at the beginning of the patching process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`uncond_sigma_end`**
    - Defines the ending value of unconditional sigma for the dynamic CFG adjustments, affecting the model's behavior at the conclusion of the patching process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`fake_uncond_sigma_end`**
    - Determines the fake ending value of unconditional sigma used in the dynamic CFG adjustments, part of the advanced settings to tweak the model's performance.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The patched model with advanced dynamic CFG settings applied, optimized for speed and efficiency.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class simpleDynamicCFGwarpDrive:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "uncond_sigma_start": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "uncond_sigma_end":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "fake_uncond_sigma_end": ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Automatic_CFG/presets"

    def patch(self, model, uncond_sigma_start, uncond_sigma_end, fake_uncond_sigma_end):
        advcfg = advancedDynamicCFG()
        print(f"                                                {Fore.CYAN}WARP DRIVE MODE ENGAGED!{Style.RESET_ALL}\n    Settings suggestions:\n"
            f"              {Fore.GREEN}1/1/1:   {Fore.YELLOW}Maaaxxxiiimum speeeeeed.{Style.RESET_ALL} {Fore.RED}Uncond disabled.{Style.RESET_ALL} {Fore.MAGENTA}Fasten your seatbelt!{Style.RESET_ALL}\n"
            f"              {Fore.GREEN}3/1/1:   {Fore.YELLOW}Risky space-time continuum distortion.{Style.RESET_ALL} {Fore.MAGENTA}Awesome for prompts with a clear subject!{Style.RESET_ALL}\n"
            f"              {Fore.GREEN}5.5/1/1: {Fore.YELLOW}Frameshift Drive Autopilot: {Fore.GREEN}Engaged.{Style.RESET_ALL} {Fore.MAGENTA}Should work with anything but do it better and faster!{Style.RESET_ALL}")

        m = advcfg.patch(model=model, automatic_cfg = "hard",
                         skip_uncond = True, uncond_sigma_start = uncond_sigma_start, uncond_sigma_end = uncond_sigma_end,
                         fake_uncond_sigma_end = fake_uncond_sigma_end, fake_uncond_sigma_start = 1000, fake_uncond_start=True,
                         fake_uncond_exp=True,fake_uncond_exp_normalize=True,fake_uncond_exp_method="previous_average",
                         cond_exp = False, cond_exp_sigma_start  = 9, cond_exp_sigma_end = uncond_sigma_start, cond_exp_method = "erf", cond_exp_normalize = True,
                         )[0]
        return (m, )

```
