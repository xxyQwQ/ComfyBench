- `LoraLoader (dirty)`: The LoraLoader node is designed to dynamically load and apply LoRA (Low-Rank Adaptation) adjustments to models and CLIP instances based on specified parameters. It facilitates the customization of model behavior and performance by integrating LoRA modifications, which can enhance or alter the model's capabilities without requiring retraining.
    - Parameters:
        - `lora_name`: The name of the LoRA file containing the adjustments to be applied. This parameter specifies which LoRA modifications will be integrated into the model and CLIP instance. Type should be `STRING`.
        - `strength_model`: The strength of the LoRA adjustments applied to the model. This parameter controls the intensity of the modifications, affecting the model's behavior. Type should be `FLOAT`.
        - `strength_clip`: The strength of the LoRA adjustments applied to the CLIP instance. This parameter controls the intensity of the modifications, affecting the CLIP's behavior. Type should be `FLOAT`.
    - Inputs:
        - `model`: The model to which LoRA adjustments will be applied. It is central to the node's operation as it determines the base model that will be modified. Type should be `MODEL`.
        - `clip`: The CLIP instance to which LoRA adjustments will be applied. This parameter allows for the customization of CLIP models alongside the primary model. Type should be `CLIP`.
    - Outputs:
        - `model`: The modified model with LoRA adjustments applied. This output reflects the integration of LoRA modifications into the original model, enhancing or altering its capabilities. Type should be `MODEL`.
        - `clip`: The modified CLIP instance with LoRA adjustments applied. This output reflects the integration of LoRA modifications into the original CLIP, enhancing or altering its capabilities. Type should be `CLIP`.