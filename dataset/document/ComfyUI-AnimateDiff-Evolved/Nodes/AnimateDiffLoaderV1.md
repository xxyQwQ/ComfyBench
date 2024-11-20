---
tags:
- AnimateDiff
- AnimateDiffContext
- Animation
---

# 🚫AnimateDiff Loader [DEPRECATED] 🎭🅐🅓
## Documentation
- Class name: `AnimateDiffLoaderV1`
- Category: ``
- Output node: `False`

This node is responsible for loading the AnimateDiff model version 1, facilitating the initialization and setup of the model for animation and image manipulation tasks. It is marked as deprecated, indicating that newer versions or methods are recommended for use. The node is deprecated, suggesting a transition to more current methodologies or versions for optimal performance.
## Input types
### Required
- **`model`**
    - Specifies the model to be loaded for animation and image manipulation tasks, central to the node's operation.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`latents`**
    - Defines the latent space parameters for the model, crucial for initializing the model's state.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
- **`model_name`**
    - Determines the specific AnimateDiff model version 1 to be loaded, affecting the animation and image manipulation capabilities.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`unlimited_area_hack`**
    - A boolean flag to enable or disable the unlimited area hack, altering the model's behavior in processing images.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`beta_schedule`**
    - Specifies the beta schedule for the model, impacting the model's training or inference behavior.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list[str]`
### Optional
- **`deprecation_warning`**
    - Provides a deprecation warning, indicating that this node is outdated and newer versions are recommended.
    - Comfy dtype: `ADEWARN`
    - Python dtype: `dict`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The loaded AnimateDiff model, ready for animation and image manipulation tasks.
    - Python dtype: `str`
- **`latent`**
    - Comfy dtype: `LATENT`
    - The initialized latent space parameters of the loaded model, essential for the model's operation.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AnimateDiffLoader_Deprecated:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latents": ("LATENT",),
                "model_name": (get_available_motion_models(),),
                "unlimited_area_hack": ("BOOLEAN", {"default": False},),
                "beta_schedule": (BetaSchedules.get_alias_list_with_first_element(BetaSchedules.SQRT_LINEAR),),
            },
            "optional": {"deprecation_warning": ("ADEWARN", {"text": "Deprecated"})},
        }

    RETURN_TYPES = ("MODEL", "LATENT")
    CATEGORY = ""
    FUNCTION = "load_mm_and_inject_params"

    def load_mm_and_inject_params(
        self,
        model: ModelPatcher,
        latents: Dict[str, torch.Tensor],
        model_name: str, unlimited_area_hack: bool, beta_schedule: str,
    ):
        # load motion module
        motion_model = load_motion_module_gen1(model_name, model)
        # get total frames
        init_frames_len = len(latents["samples"])  # deprecated - no longer used for anything lol
        # set injection params
        params = InjectionParams(
                unlimited_area_hack=unlimited_area_hack,
                apply_mm_groupnorm_hack=True,
                model_name=model_name,
                apply_v2_properly=False,
        )
        # inject for use in sampling code
        model = ModelPatcherAndInjector.create_from(model, hooks_only=True)
        model.motion_models = MotionModelGroup(motion_model)
        model.motion_injection_params = params

        # save model sampling from BetaSchedule as object patch
        # if autoselect, get suggested beta_schedule from motion model
        if beta_schedule == BetaSchedules.AUTOSELECT and not model.motion_models.is_empty():
            beta_schedule = model.motion_models[0].model.get_best_beta_schedule(log=True)
        new_model_sampling = BetaSchedules.to_model_sampling(beta_schedule, model)
        if new_model_sampling is not None:
            model.add_object_patch("model_sampling", new_model_sampling)

        del motion_model
        return (model, latents)

```
