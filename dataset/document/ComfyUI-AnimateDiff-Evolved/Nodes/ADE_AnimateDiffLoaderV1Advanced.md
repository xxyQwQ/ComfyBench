---
tags:
- AnimateDiff
- AnimateDiffContext
- Animation
---

# 🚫AnimateDiff Loader (Advanced) [DEPRECATED] 🎭🅐🅓
## Documentation
- Class name: `ADE_AnimateDiffLoaderV1Advanced`
- Category: ``
- Output node: `False`

This node is designed for advanced loading of AnimateDiff models, offering enhanced configuration options and optimizations for deprecated model versions. It aims to provide a more flexible and efficient way to work with older AnimateDiff models, ensuring compatibility and performance.
## Input types
### Required
- **`model`**
    - Specifies the AnimateDiff model to be loaded, allowing for advanced configuration and optimization of deprecated models.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`latents`**
    - Defines the latent variables associated with the AnimateDiff model, crucial for its operation and performance.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
- **`model_name`**
    - Identifies the specific AnimateDiff model to be loaded, enabling precise selection and configuration.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`unlimited_area_hack`**
    - A boolean flag that, when enabled, allows for bypassing certain limitations, enhancing the model's flexibility.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`context_length`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`context_stride`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`context_overlap`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`context_schedule`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`closed_loop`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`beta_schedule`**
    - Determines the beta schedule used during the model's operation, affecting its performance and outcomes.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`deprecation_warning`**
    - Provides a warning about the deprecated status of the model, advising on its experimental or limited use.
    - Comfy dtype: `ADEWARN`
    - Python dtype: `str`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - Outputs the loaded AnimateDiff model, configured and optimized for use.
    - Python dtype: `AnimateDiffModel`
- **`latent`**
    - Comfy dtype: `LATENT`
    - unknown
    - Python dtype: `unknown`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AnimateDiffLoaderAdvanced_Deprecated:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latents": ("LATENT",),
                "model_name": (get_available_motion_models(),),
                "unlimited_area_hack": ("BOOLEAN", {"default": False},),
                "context_length": ("INT", {"default": 16, "min": 0, "max": 1000}),
                "context_stride": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "context_overlap": ("INT", {"default": 4, "min": 0, "max": 1000}),
                "context_schedule": (ContextSchedules.LEGACY_UNIFORM_SCHEDULE_LIST,),
                "closed_loop": ("BOOLEAN", {"default": False},),
                "beta_schedule": (BetaSchedules.get_alias_list_with_first_element(BetaSchedules.SQRT_LINEAR),),
            },
            "optional": {"deprecation_warning": ("ADEWARN", {"text": "Deprecated"})},
        }

    RETURN_TYPES = ("MODEL", "LATENT")
    CATEGORY = ""
    FUNCTION = "load_mm_and_inject_params"

    def load_mm_and_inject_params(self,
            model: ModelPatcher,
            latents: Dict[str, torch.Tensor],
            model_name: str, unlimited_area_hack: bool,
            context_length: int, context_stride: int, context_overlap: int, context_schedule: str, closed_loop: bool,
            beta_schedule: str,
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
        context_group = ContextOptionsGroup()
        context_group.add(
            ContextOptions(
                context_length=context_length,
                context_stride=context_stride,
                context_overlap=context_overlap,
                context_schedule=context_schedule,
                closed_loop=closed_loop,
                )
            )
        # set context settings
        params.set_context(context_options=context_group)
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
