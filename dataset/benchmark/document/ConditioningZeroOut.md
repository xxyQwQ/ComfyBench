- `ConditioningZeroOut`: This node zeroes out specific elements within the conditioning data structure, effectively neutralizing their influence in subsequent processing steps. It's designed for advanced conditioning operations where direct manipulation of the conditioning's internal representation is required.
    - Inputs:
        - `conditioning` (Required): The conditioning data structure to be modified. This node zeroes out the 'pooled_output' elements within each conditioning entry, if present. Type should be `CONDITIONING`.
    - Outputs:
        - `conditioning`: The modified conditioning data structure, with 'pooled_output' elements set to zero where applicable. Type should be `CONDITIONING`.