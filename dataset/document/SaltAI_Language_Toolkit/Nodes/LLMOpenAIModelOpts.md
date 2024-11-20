---
tags:
- LoRA
---

# ∞ Model Options
## Documentation
- Class name: `LLMOpenAIModelOpts`
- Category: `SALT/Language Toolkit/Loaders/Options`
- Output node: `False`

The LLMOpenAIModelOpts node is designed to configure and customize the options for language models and their embedding counterparts. It allows for the adjustment of various parameters such as temperature, retries, and embedding dimensions, enabling fine-tuning of the model's behavior and performance.
## Input types
### Required
- **`llm_model`**
    - The 'llm_model' parameter is a dictionary containing the language model and embedding model to be configured. It is crucial for specifying the models whose options are to be set, affecting the overall execution and results of the node.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `Dict[str, Any]`
### Optional
- **`model_temperature`**
    - Specifies the temperature setting for the language model, influencing its creativity and randomness.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`model_max_tokens`**
    - Defines the maximum number of tokens the language model can generate, affecting the length of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`model_api_max_retries`**
    - Determines the number of times the API will retry a request in case of failure, ensuring reliability.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`model_api_timeout`**
    - Sets the timeout duration for API requests, impacting the response time.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`model_reuse_anyscale_client`**
    - Indicates whether to reuse the Anyscale client across requests, optimizing resource usage.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`multimodal_max_new_tokens`**
    - Specifies the maximum number of new tokens for multimodal inputs, affecting the output length.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`multimodal_image_detail`**
    - Determines the level of detail for images in multimodal inputs, influencing the model's focus.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`embed_batch_size`**
    - Sets the batch size for embedding model operations, affecting performance and resource utilization.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`embed_dimensions`**
    - Specifies the dimensionality of the embeddings, influencing the detail level of the generated embeddings.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`embed_api_max_retries`**
    - Determines the number of times the embedding API will retry in case of failure, ensuring reliability.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`embed_api_timeout`**
    - Sets the timeout duration for embedding API requests, impacting the response time.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`embed_reuse_anyscale_client`**
    - Indicates whether to reuse the Anyscale client for embedding operations, optimizing resource usage.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`model_additional_kwargs`**
    - Allows for the specification of additional keyword arguments for the model, enabling further customization.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`embed_additional_kwargs`**
    - Allows for the specification of additional keyword arguments for the embedding model, enabling further customization.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`model_system_prompt`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
## Output types
- **`model`**
    - Comfy dtype: `LLM_MODEL`
    - The 'model' output is a dictionary reflecting the updated configurations of the language model and embedding model. It encapsulates the adjustments made to the models' options, influencing their operational characteristics and output.
    - Python dtype: `Dict[str, Any]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMOpenAIModelOpts:
    """
    Sets various options for the model, and embedding.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
            },
            "optional": {
                "model_temperature": ("FLOAT", {"min": 0, "max": 1.0, "default": DEFAULT_TEMPERATURE, "step": 0.001}),
                "model_max_tokens": ("INT", {"min": 8, "default": 4096}),
                "model_api_max_retries": ("INT", {"min": 1, "max": 12, "default": 3}),
                "model_api_timeout": ("INT", {"min": 8, "max": 120, "default": 60}),
                "model_reuse_anyscale_client": ("BOOLEAN", {"default": True}),
                
                "multimodal_max_new_tokens": ("INT", {"min": 8, "default": 300}),
                "multimodal_image_detail": (["low", "high", "auto"],),

                "embed_batch_size": ("INT", {"min": 8, "default": DEFAULT_EMBED_BATCH_SIZE}),
                "embed_dimensions": ("INT", {"min": 1, "default": DEFAULT_EMBEDDING_DIM}),
                "embed_api_max_retries": ("INT", {"min": 1, "max": 12, "default": 3}),
                "embed_api_timeout": ("INT", {"min": 8, "max": 120, "default": 60}),
                "embed_reuse_anyscale_client": ("BOOLEAN", {"default": True}),
                
                "model_additional_kwargs": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "", "placeholder": "Additional model kwargs JSON"}),
                "embed_additional_kwargs": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "", "placeholder": "Additional embed kwargs JSON"}),
                "model_system_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "System directions, or rules to follow globally across nodes."}),
            }
        }

    RETURN_TYPES = ("LLM_MODEL", )
    RETURN_NAMES = ("model", )

    FUNCTION = "set_options"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Loaders/Options"

    def set_options(self, llm_model:Dict[str, Any], **kwargs) -> Dict[str, Any]:
        llm = llm_model['llm']
        embed = llm_model['embed_model']
        
        # LLM Options
        llm.temperature = kwargs.get("model_temperature", DEFAULT_TEMPERATURE)
        llm.max_retries = kwargs.get("model_api_max_retries", 3)
        llm.reuse_client = kwargs.get("model_reuse_anyscale_client", True)
        llm.additional_kwargs = json.loads(kwargs.get("model_additional_kwargs", {}).strip()) if kwargs.get("model_additional_kwargs", {}).strip() != "" else {} # Default to `None` if empty string
        llm.system_prompt = kwargs.get("model_system_prompt", None)

        # Embed Options
        embed.embed_batch_size = kwargs.get("embed_batch_size", DEFAULT_EMBED_BATCH_SIZE)
        embed.dimensions = kwargs.get("embed_dimensions", DEFAULT_EMBEDDING_DIM) if kwargs.get("embed_dimensions", DEFAULT_EMBEDDING_DIM) > 0 else None # Default to `None` if not above 0
        embed.additional_kwargs = json.loads(kwargs.get("embed_additional_kwargs", {}).strip()) if kwargs.get("embed_additional_kwargs", "").strip() != "" else {} # Default to `None` if empty string
        embed.max_retries = kwargs.get("embed_api_max_retries", 3)
        embed.timeout = kwargs.get("embed_api_timeout", 60)
        embed.reuse_client = kwargs.get("embed_reuse_anyscale_client", True)

        if isinstance(llm, OpenAIMultiModal):
            llm.max_new_tokens = kwargs.get("multimodal_max_new_tokens", 300)
            llm.image_detail = kwargs.get("multimodal_image_detail", "low")
        else:
            llm.max_tokens = kwargs.get("model_max_tokens", 4096)

        llm_model['llm'] = llm
        llm_model['embed_model'] = embed

        return (llm_model,)

```
