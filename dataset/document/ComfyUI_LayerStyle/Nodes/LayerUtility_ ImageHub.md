# LayerUtility: ImageHub
## Documentation
- Class name: `LayerUtility: ImageHub`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

The ImageHub node serves as a dynamic selector and router for image and mask inputs, allowing for the selection of specific image-mask pairs based on a specified output channel. It facilitates the flexible manipulation and routing of image data within a node network, enabling the creation of complex image processing pipelines.
## Input types
### Required
- **`output`**
    - Specifies the output channel to select the corresponding image and mask inputs. It determines which set of inputs (image and mask) will be used for the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`random_output`**
    - A boolean flag that, if set, could potentially alter the selection logic for choosing the output image and mask, adding an element of randomness to the output selection process.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`input1_image`**
    - The first image input option for the node, which can be selected as the output based on the 'output' parameter value.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`input1_mask`**
    - The mask associated with the first image input, providing an option for output selection alongside the corresponding image.
    - Comfy dtype: `MASK`
    - Python dtype: `Mask`
- **`input2_image`**
    - The second image input option, selectable based on the output parameter.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`input2_mask`**
    - The mask associated with the second image input, used in conjunction with the corresponding image for output selection.
    - Comfy dtype: `MASK`
    - Python dtype: `Mask`
- **`input3_image`**
    - The third image input option, which can be chosen based on the specified output channel.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`input3_mask`**
    - The mask for the third image input, available for selection with its corresponding image based on the output parameter.
    - Comfy dtype: `MASK`
    - Python dtype: `Mask`
- **`input4_image`**
    - The fourth image input option, selectable for output based on the specified channel.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`input4_mask`**
    - The mask associated with the fourth image input, selectable alongside the corresponding image.
    - Comfy dtype: `MASK`
    - Python dtype: `Mask`
- **`input5_image`**
    - The fifth image input option, which can be selected for output based on the specified channel.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`input5_mask`**
    - The mask for the fifth image input, available for selection with its corresponding image.
    - Comfy dtype: `MASK`
    - Python dtype: `Mask`
- **`input6_image`**
    - The sixth image input option, selectable for output based on the output parameter.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`input6_mask`**
    - The mask associated with the sixth image input, used in output selection alongside the corresponding image.
    - Comfy dtype: `MASK`
    - Python dtype: `Mask`
- **`input7_image`**
    - The seventh image input option, which can be chosen for output based on the specified channel.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`input7_mask`**
    - The mask for the seventh image input, available for selection with its corresponding image.
    - Comfy dtype: `MASK`
    - Python dtype: `Mask`
- **`input8_image`**
    - The eighth image input option, selectable for output based on the output parameter.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`input8_mask`**
    - The mask associated with the eighth image input, used in output selection alongside the corresponding image.
    - Comfy dtype: `MASK`
    - Python dtype: `Mask`
- **`input9_image`**
    - The ninth and final image input option, which can be selected for output based on the specified channel.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`input9_mask`**
    - The mask for the ninth image input, available for selection with its corresponding image.
    - Comfy dtype: `MASK`
    - Python dtype: `Mask`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The selected image output based on the specified output channel and input parameters.
    - Python dtype: `Image`
- **`mask`**
    - Comfy dtype: `MASK`
    - The selected mask output corresponding to the chosen image, based on the specified output channel and input parameters.
    - Python dtype: `Mask`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ImageHub:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "output":  ("INT", {"default": 1, "min": 1, "max": 9, "step": 1}),
                "random_output": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "input1_image": ("IMAGE",),
                "input1_mask": ("MASK",),
                "input2_image": ("IMAGE",),
                "input2_mask": ("MASK",),
                "input3_image": ("IMAGE",),
                "input3_mask": ("MASK",),
                "input4_image": ("IMAGE",),
                "input4_mask": ("MASK",),
                "input5_image": ("IMAGE",),
                "input5_mask": ("MASK",),
                "input6_image": ("IMAGE",),
                "input6_mask": ("MASK",),
                "input7_image": ("IMAGE",),
                "input7_mask": ("MASK",),
                "input8_image": ("IMAGE",),
                "input8_mask": ("MASK",),
                "input9_image": ("IMAGE",),
                "input9_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask")
    FUNCTION = 'image_hub'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_hub(self, output, random_output,
                  input1_image=None, input1_mask=None,
                  input2_image=None, input2_mask=None,
                  input3_image=None, input3_mask=None,
                  input4_image=None, input4_mask=None,
                  input5_image=None, input5_mask=None,
                  input6_image=None, input6_mask=None,
                  input7_image=None, input7_mask=None,
                  input8_image=None, input8_mask=None,
                  input9_image=None, input9_mask=None,
                  ):

        output_list = []
        if input1_image is not None or input1_mask is not None:
            output_list.append(1)
        if input2_image is not None or input2_mask is not None:
            output_list.append(2)
        if input3_image is not None or input3_mask is not None:
            output_list.append(3)
        if input4_image is not None or input4_mask is not None:
            output_list.append(4)
        if input5_image is not None or input5_mask is not None:
            output_list.append(5)
        if input6_image is not None or input6_mask is not None:
            output_list.append(6)
        if input7_image is not None or input7_mask is not None:
            output_list.append(7)
        if input8_image is not None or input8_mask is not None:
            output_list.append(8)
        if input9_image is not None or input9_mask is not None:
            output_list.append(9)

        log(f"output_list={output_list}")
        if len(output_list) == 0:
            log(f"{NODE_NAME} is skip, because No Input.", message_type='error')
            return (None, None)

        if random_output:
            index = random.randint(1, len(output_list))
            output = output_list[index - 1]

        ret_image = None
        ret_mask = None
        if output == 1:
            if input1_image is not None:
                ret_image = input1_image
            if input1_mask is not None:
                ret_mask = input1_mask
        elif output == 2:
            if input2_image is not None:
                ret_image = input2_image
            if input2_mask is not None:
                ret_mask = input2_mask
        elif output == 3:
            if input3_image is not None:
                ret_image = input3_image
            if input3_mask is not None:
                ret_mask = input3_mask
        elif output == 4:
            if input4_image is not None:
                ret_image = input4_image
            if input4_mask is not None:
                ret_mask = input4_mask
        elif output == 5:
            if input5_image is not None:
                ret_image = input5_image
            if input5_mask is not None:
                ret_mask = input5_mask
        elif output == 6:
            if input6_image is not None:
                ret_image = input6_image
            if input6_mask is not None:
                ret_mask = input6_mask
        elif output == 7:
            if input7_image is not None:
                ret_image = input7_image
            if input7_mask is not None:
                ret_mask = input7_mask
        elif output == 8:
            if input8_image is not None:
                ret_image = input8_image
            if input8_mask is not None:
                ret_mask = input8_mask
        else:
            if input9_image is not None:
                ret_image = input9_image
            if input9_mask is not None:
                ret_mask = input9_mask

        if ret_image is None and ret_mask is None:
            log(f"{NODE_NAME} have {output_list} inputs, output is {output}, but there is no corresponding input.", message_type="error")
        elif ret_image is None:
            log(f"{NODE_NAME} have {output_list} inputs, output is {output}, but image is None.", message_type='finish')
        elif ret_mask is None:
            log(f"{NODE_NAME} have {output_list} inputs, output is {output}, but mask is None.", message_type='finish')
        else:
            log(f"{NODE_NAME} have {output_list} inputs, output is {output}.", message_type='finish')

        return (ret_image, ret_mask)

```
