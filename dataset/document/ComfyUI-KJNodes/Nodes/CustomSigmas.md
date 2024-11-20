---
tags:
- SigmaScheduling
---

# Custom Sigmas
## Documentation
- Class name: `CustomSigmas`
- Category: `KJNodes/noise`
- Output node: `False`

The CustomSigmas node is designed to transform a list of float values into a tensor of sigmas, facilitating the manipulation and application of noise levels in generative models. It abstracts the process of converting numerical lists into a structured format that models can interpret for noise injection or adjustment.
## Input types
### Required
- **`sigmas_string`**
    - A string of comma-separated values representing sigma levels, which are converted into a tensor. This input is essential for defining the noise characteristics to be applied in generative processes.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`interpolate_to_steps`**
    - An integer specifying the number of steps to which the sigma values should be interpolated. This parameter adjusts the length of the sigma tensor to match the desired number of steps in the generative process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`SIGMAS`**
    - Comfy dtype: `SIGMAS`
    - A tensor of sigma values derived from the input list of floats, ready for use in noise application or adjustment within generative models.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CustomSigmas:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                     "sigmas_string" :("STRING", {"default": "14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029","multiline": True}),
                     "interpolate_to_steps": ("INT", {"default": 10,"min": 0, "max": 255, "step": 1}),
                     }
                }
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("SIGMAS",)
    CATEGORY = "KJNodes/noise"
    FUNCTION = "customsigmas"
    DESCRIPTION = """
Creates a sigmas tensor from a string of comma separated values.  
Examples: 
   
Nvidia's optimized AYS 10 step schedule for SD 1.5:  
14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029  
SDXL:   
14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029  
SVD:  
700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.002  
"""
    def customsigmas(self, sigmas_string, interpolate_to_steps):
        sigmas_list = sigmas_string.split(', ')
        sigmas_float_list = [float(sigma) for sigma in sigmas_list]
        sigmas_tensor = torch.FloatTensor(sigmas_float_list)
        if len(sigmas_tensor) != interpolate_to_steps + 1:
            sigmas_tensor = self.loglinear_interp(sigmas_tensor, interpolate_to_steps + 1)
        sigmas_tensor[-1] = 0
        return (sigmas_tensor.float(),)
     
    def loglinear_interp(self, t_steps, num_steps):
        """
        Performs log-linear interpolation of a given array of decreasing numbers.
        """
        t_steps_np = t_steps.numpy()

        xs = np.linspace(0, 1, len(t_steps_np))
        ys = np.log(t_steps_np[::-1])
        
        new_xs = np.linspace(0, 1, num_steps)
        new_ys = np.interp(new_xs, xs, ys)
        
        interped_ys = np.exp(new_ys)[::-1].copy()
        interped_ys_tensor = torch.tensor(interped_ys)
        return interped_ys_tensor

```
