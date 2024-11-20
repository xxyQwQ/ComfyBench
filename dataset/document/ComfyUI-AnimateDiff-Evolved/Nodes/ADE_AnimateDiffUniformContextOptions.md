---
tags:
- AnimateDiff
- AnimateDiffContext
- Animation
---

# Context Options◆Looped Uniform 🎭🅐🅓
## Documentation
- Class name: `ADE_AnimateDiffUniformContextOptions`
- Category: ``
- Output node: `False`

This node is designed to provide uniform context options for the AnimateDiff process, facilitating the generation of animations with consistent context across frames. It is marked as legacy, indicating that it may have been superseded by newer methods or nodes for managing context in animation generation.
## Input types
### Required
- **`context_length`**
    - Defines the length of the context used in the animation process, influencing the amount of information considered for generating each frame.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`context_stride`**
    - Determines the stride between contexts in the animation, affecting the transition and flow between frames.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`context_overlap`**
    - Specifies the degree of overlap between contexts of consecutive frames, affecting the smoothness and continuity of the animation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`context_schedule`**
    - Specifies the scheduling method for context application, influencing the timing and sequence of animation frames.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `ContextSchedules`
- **`closed_loop`**
    - Indicates whether the animation should loop back to the beginning after completing, creating a continuous loop effect.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`fuse_method`**
    - Determines the method used to fuse multiple contexts together, impacting the overall coherence and flow of the generated animation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `ContextFuseMethod`
- **`use_on_equal_length`**
    - A boolean flag indicating whether the context options should be applied uniformly across frames of equal length, ensuring consistent animation characteristics.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`start_percent`**
    - Indicates the starting point of the animation as a percentage, allowing for precise control over the animation's initiation within the context.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`guarantee_steps`**
    - Guarantees a minimum number of steps in the animation, ensuring that the animation reaches a certain length regardless of other parameters.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`prev_context`**
    - The previous context options group, which can be used as a basis for generating the new context options, facilitating continuity across animations.
    - Comfy dtype: `CONTEXT_OPTIONS`
    - Python dtype: `ContextOptionsGroup`
- **`view_opts`**
    - View options that specify additional parameters for rendering the animation, further customizing the output.
    - Comfy dtype: `VIEW_OPTS`
    - Python dtype: `ContextOptions`
- **`deprecation_warning`**
    - A notice that this node is considered legacy and may be replaced by newer methods or nodes in future updates.
    - Comfy dtype: `ADEWARN`
    - Python dtype: `str`
## Output types
- **`CONTEXT_OPTS`**
    - Comfy dtype: `CONTEXT_OPTIONS`
    - The generated context options, ready to be utilized in the AnimateDiff process for creating uniform animations.
    - Python dtype: `ContextOptionsGroup`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [ADE_AnimateDiffLoaderWithContext](../../ComfyUI-AnimateDiff-Evolved/Nodes/ADE_AnimateDiffLoaderWithContext.md)



## Source code
```python
class LegacyLoopedUniformContextOptionsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context_length": ("INT", {"default": 16, "min": 1, "max": LENGTH_MAX}),
                "context_stride": ("INT", {"default": 1, "min": 1, "max": STRIDE_MAX}),
                "context_overlap": ("INT", {"default": 4, "min": 0, "max": OVERLAP_MAX}),
                "context_schedule": (ContextSchedules.LEGACY_UNIFORM_SCHEDULE_LIST,),
                "closed_loop": ("BOOLEAN", {"default": False},),
                #"sync_context_to_pe": ("BOOLEAN", {"default": False},),
            },
            "optional": {
                "fuse_method": (ContextFuseMethod.LIST, {"default": ContextFuseMethod.FLAT}),
                "use_on_equal_length": ("BOOLEAN", {"default": False},),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
                "prev_context": ("CONTEXT_OPTIONS",),
                "view_opts": ("VIEW_OPTS",),
                "deprecation_warning": ("ADEWARN", {"text": ""}),
            }
        }
    
    RETURN_TYPES = ("CONTEXT_OPTIONS",)
    RETURN_NAMES = ("CONTEXT_OPTS",)
    CATEGORY = ""  # No Category, so will not appear in menu
    FUNCTION = "create_options"

    def create_options(self, fuse_method: str=ContextFuseMethod.FLAT, context_schedule: str=None, **kwargs):
        return LoopedUniformContextOptionsNode.create_options(self, fuse_method=fuse_method, **kwargs)

```
