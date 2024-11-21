- `ImpactWildcardEncode`: The ImpactWildcardEncode node is designed to dynamically encode text by replacing specified wildcard patterns with corresponding values or options. It leverages a comprehensive wildcard system to interpret and transform text inputs based on predefined or custom wildcard dictionaries, supporting complex pattern matching and replacement strategies to tailor text outputs for varied applications.
    - Inputs:
        - `model` (Required): This input specifies the model to be used in the encoding process, enabling the node to leverage specific model capabilities for text transformation. Type should be `MODEL`.
        - `clip` (Required): The 'clip' parameter represents the CLIP model used for encoding, providing a mechanism for integrating visual context into the text transformation process. Type should be `CLIP`.
        - `wildcard_text` (Required): This input contains the text with wildcard patterns that the node will interpret and replace, serving as the primary source for the encoding operation. Type should be `STRING`.
        - `populated_text` (Required): The text resulting from the wildcard replacement process, which is further processed or utilized within the node's workflow. Type should be `STRING`.
        - `mode` (Required): A boolean input that toggles between different modes of operation, such as 'Populate' or 'Fixed', affecting how text is processed and wildcards are handled. Type should be `BOOLEAN`.
        - `Select to add LoRA` (Required): Allows selection of a LoRA to add to the text, introducing specific logic or rules-based adjustments to the encoding process. Type should be `COMBO[STRING]`.
        - `Select to add Wildcard` (Required): Enables the selection of additional wildcards to be added to the text, expanding the node's capability to transform and customize text outputs. Type should be `COMBO[STRING]`.
        - `seed` (Required): A numerical input that seeds the randomization process, ensuring reproducibility and consistency in the text transformation outcomes. Type should be `INT`.
    - Outputs:
        - `model`: The model output, potentially modified or utilized within the node's processing flow. Type should be `MODEL`.
        - `clip`: The CLIP model output, reflecting any changes or usage within the node's operations. Type should be `CLIP`.
        - `conditioning`: Conditioning information generated or modified by the node, used in further processing steps. Type should be `CONDITIONING`.
        - `populated_text`: The final text output, with all specified wildcards replaced and processed according to the node's logic and parameters. Type should be `STRING`.