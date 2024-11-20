# DownloadAndLoadFlorence2Model
## Documentation
- Class name: `DownloadAndLoadFlorence2Model`
- Category: `Florence2`
- Output node: `False`

This node is designed to download and load a specified Florence2 model, applying the chosen precision and attention mechanism settings. It ensures the model is ready for use by initializing it with the appropriate device, data type, and attention implementation.
## Input types
### Required
- **`model`**
    - Specifies the repository ID of the Florence2 model to be downloaded and loaded. It influences the model's capabilities and features.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`precision`**
    - Determines the numerical precision (e.g., bf16, fp16, fp32) of the model's computations, affecting performance and memory usage.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`attention`**
    - Selects the attention mechanism implementation to be used by the model, impacting its performance and accuracy.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`florence2_model`**
    - Comfy dtype: `FL2MODEL`
    - A dictionary containing the loaded model, its processor, and the data type used for computations.
    - Python dtype: `Dict[str, Any]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class DownloadAndLoadFlorence2Model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [ 
                    'microsoft/Florence-2-base',
                    'microsoft/Florence-2-base-ft',
                    'microsoft/Florence-2-large',
                    'microsoft/Florence-2-large-ft',
                    'HuggingFaceM4/Florence-2-DocVQA'
                    ],
                    {
                    "default": 'microsoft/Florence-2-base'
                    }),
            "precision": ([ 'fp16','bf16','fp32'],
                    {
                    "default": 'fp16'
                    }),
            "attention": (
                    [ 'flash_attention_2', 'sdpa', 'eager'],
                    {
                    "default": 'sdpa'
                    }),

            },
        }

    RETURN_TYPES = ("FL2MODEL",)
    RETURN_NAMES = ("florence2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Florence2"

    def loadmodel(self, model, precision, attention):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        model_name = model.rsplit('/', 1)[-1]
        model_path = os.path.join(folder_paths.models_dir, "LLM", model_name)
        
        if not os.path.exists(model_path):
            print(f"Downloading Lumina model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model,
                            local_dir=model_path,
                            local_dir_use_symlinks=False)
            
        print(f"using {attention} for attention")
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): #workaround for unnecessary flash_attn requirement
            model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation=attention, device_map=device, torch_dtype=dtype,trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        florence2_model = {
            'model': model, 
            'processor': processor,
            'dtype': dtype
            }

        return (florence2_model,)

```
