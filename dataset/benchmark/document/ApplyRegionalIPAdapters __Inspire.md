- `ApplyRegionalIPAdapters __Inspire`: This node is designed to apply a series of regional IP adapter transformations to a model, leveraging a pipeline of IP adapters and regional IP adapters to enhance or modify the model's capabilities based on regional information.
    - Inputs:
        - `ipadapter_pipe` (Required): A tuple containing the main IP adapter, model, clip vision module, insightface module, and lora loader, which collectively form the pipeline through which the regional IP adapters are applied. Type should be `IPADAPTER_PIPE`.
        - `regional_ipadapter1` (Required): A regional IP adapter that is applied to the model as part of the transformation process. Type should be `REGIONAL_IPADAPTER`.
    - Outputs:
        - `model`: The transformed model after applying the regional IP adapters. Type should be `MODEL`.
