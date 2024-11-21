- `AppendInstanceDiffusionTracking`: The AppendInstanceDiffusionTracking node is designed for integrating and enhancing tracking data within the InstanceDiffusion framework. It merges tracking information from two sources, ensuring that class data is combined without duplication, and concatenates prompts for enriched context. This node facilitates the creation of comprehensive tracking datasets for InstanceDiffusion applications, streamlining the process of data preparation for instance-based diffusion tasks.
    - Inputs:
        - `tracking_i` (Required): unknown Type should be `TRACKING`.
        - `prompt_i` (Optional): unknown Type should be `STRING`.
    - Outputs:
        - `tracking`: The merged tracking data, combining class information and IDs from both input sources without duplication. Type should be `TRACKING`.
        - `prompt`: A concatenated string of the two input prompts, providing a unified context for the diffusion task. Type should be `STRING`.