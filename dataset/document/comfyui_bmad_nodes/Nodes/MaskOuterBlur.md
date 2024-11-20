---
tags:
- Blur
- VisualEffects
---

# MaskOuterBlur
## Documentation
- Class name: `MaskOuterBlur`
- Category: `Bmad/CV/Misc`
- Output node: `False`

The MaskOuterBlur node is designed to selectively apply a blur effect to the outer regions of an image, utilizing a mask to distinguish between areas to be blurred and those to remain sharp. This technique is aimed at enhancing focus on specific parts of an image or creating a depth-of-field effect, making it particularly useful for image editing and post-processing applications.
## Input types
### Required
- **`src`**
    - The source image to which the blur effect will be applied. It serves as the primary input for determining the areas to be blurred based on the mask.
    - Comfy dtype: `IMAGE`
    - Python dtype: `numpy.ndarray`
- **`mask`**
    - The mask used to differentiate between the areas of the image to be blurred and those to remain sharp. It plays a crucial role in guiding the selective blurring process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `numpy.ndarray`
- **`kernel_size`**
    - Specifies the size of the blur kernel. A larger kernel size results in a more pronounced blur effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`paste_src`**
    - A boolean flag indicating whether the original source image should be pasted over the blurred areas, preserving detail in the non-blurred regions.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image after applying the selective blur effect, showcasing the enhanced focus or depth-of-field effect.
    - Python dtype: `numpy.ndarray`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MaskOuterBlur:  # great, another "funny" name; what would you call this?
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src": ("IMAGE",),
                "mask": ("IMAGE",),
                "kernel_size": ("INT", {"default": 16, "min": 2, "max": 150, "step": 2}),
                "paste_src": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compute"
    CATEGORY = f"{cv_category_path}/Misc"

    def compute(self, src, mask, kernel_size, paste_src):
        from comfy.model_management import is_nvidia

        # setup input data
        kernel_size += 1
        image = tensor2opencv(src, 3)
        mask = tensor2opencv(mask, 1)

        # setup kernel ( maybe add optional input later for custom kernel? )
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        kernel = np.where(kernel > 0, 1, 0)

        #  resize mask if it's size doesn't match the image's
        if image.shape[0:2] != mask.shape[0:2]:
            print("MaskedOuterBlur node will resize mask to fit the image.")
            mask = cv.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv.INTER_LINEAR)

        # extend image borders so that the algorithm doesn't have to take them into account
        border_size = kernel_size // 2
        image_extended = cv.copyMakeBorder(image, border_size, border_size, border_size, border_size,
                                           cv.BORDER_REPLICATE)
        mask_extended = cv.copyMakeBorder(mask, border_size, border_size, border_size, border_size, cv.BORDER_REPLICATE)

        # convert the image to float32
        image_float32 = image_extended.astype('float32')
        mask_float32 = mask_extended.astype('float32')

        if is_nvidia():  # is this check legit?
            import cupy as cp
            from numba.cuda import get_current_device
            from .utils.mask_outer_blur import blur_cuda

            # setup cupy arrays
            image_cupy = cp.asarray(image_float32)
            mask_cupy = cp.asarray(mask_float32)
            # note: don't pass extended size here; more data than needed to retrieve from gpu.
            #       instead, rawkernel outputs the final directly with the kernel size in mind
            #       and there is no need to crop after the computation
            out = cp.zeros((mask.shape[0], mask.shape[1], 3), dtype=cp.float32)
            kernel_gpu = cp.asarray(kernel)

            # setup grid/block sizes
            gpu = get_current_device()
            w, h = mask.shape[1], mask.shape[0]
            block_dim_x, block_dim_y = np.floor(
                np.array([w / h, h / w]) * gpu.MAX_THREADS_PER_BLOCK ** (1 / 2)).astype(np.int32)
            gridx, gridy = np.ceil(np.array([w / block_dim_x, h / block_dim_y])).astype(np.int32)

            # run on gpu, and then fetch result as numpy array
            blur_cuda((gridx, gridy), (block_dim_x, block_dim_y),
                      (image_cupy, mask_cupy, out, kernel_gpu, kernel_size, mask_extended.shape[1],
                       mask_extended.shape[0]))
            result_float32 = cp.asnumpy(out)

        else:  # run on cpu
            from .utils.mask_outer_blur import blur_cpu
            result_float32 = blur_cpu(image_float32, mask_float32, kernel, kernel_size, mask_extended.shape[1],
                                      mask_extended.shape[0])
            # remove added borders ( this is not required in gpu version;
            #                        only done here to avoid computing two sets of coordinates for every pixel )
            result_float32 = result_float32[border_size:-border_size, border_size:-border_size, :]

        if paste_src:  # paste src onto result using mask
            indices = mask > 0
            result_float32[indices, :] = image[indices, :]

        result = opencv2tensor(result_float32)
        return (result,)

```
