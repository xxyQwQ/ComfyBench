# SparseCtrl Weight Extras 🛂🅐🅒🅝
## Documentation
- Class name: `ACN_SparseCtrlWeightExtras`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl/extras`
- Output node: `False`

This node is designed to create and manage additional weight parameters for Sparse Control Networks, enabling fine-tuned control over the network's behavior by adjusting hint, non-hint, and mask multipliers.
## Input types
### Optional
- **`cn_extras`**
    - A dictionary to store extra weight parameters, allowing for the customization and fine-tuning of the Sparse Control Network's behavior.
    - Comfy dtype: `CN_WEIGHTS_EXTRAS`
    - Python dtype: `dict[str]`
- **`sparse_hint_mult`**
    - Multiplier for sparse hints, influencing how strongly hints affect the network's output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sparse_nonhint_mult`**
    - Multiplier for non-hints, adjusting the influence of non-hint areas on the network's output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sparse_mask_mult`**
    - Multiplier for the mask, determining the impact of the mask on the network's behavior.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`autosize`**
    - An optional parameter to adjust the size of the network's components, providing flexibility in network configuration and optimization.
    - Comfy dtype: `ACNAUTOSIZE`
    - Python dtype: `{'padding': int}`
## Output types
- **`cn_extras`**
    - Comfy dtype: `CN_WEIGHTS_EXTRAS`
    - Updated dictionary of extra weight parameters, reflecting the adjustments made for hint, non-hint, and mask multipliers.
    - Python dtype: `dict[str]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SparseWeightExtras:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "cn_extras": ("CN_WEIGHTS_EXTRAS",),
                "sparse_hint_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "sparse_nonhint_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "sparse_mask_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "autosize": ("ACNAUTOSIZE", {"padding": 50}),
            }
        }
    
    RETURN_TYPES = ("CN_WEIGHTS_EXTRAS", )
    RETURN_NAMES = ("cn_extras", )
    FUNCTION = "create_weight_extras"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl/extras"

    def create_weight_extras(self, cn_extras: dict[str]={}, sparse_hint_mult=1.0, sparse_nonhint_mult=1.0, sparse_mask_mult=1.0):
        cn_extras = cn_extras.copy()
        cn_extras[SparseConst.HINT_MULT] = sparse_hint_mult
        cn_extras[SparseConst.NONHINT_MULT] = sparse_nonhint_mult
        cn_extras[SparseConst.MASK_MULT] = sparse_mask_mult
        return (cn_extras, )

```
