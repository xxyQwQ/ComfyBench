- `FromIPAdapterPipe __Inspire`: The `FromIPAdapterPipe` node is designed to decompose a previously constructed IP adapter pipeline into its constituent components. This node facilitates the retrieval of individual elements such as the IP adapter, model, and additional features like CLIP vision and InsightFace from a bundled pipeline, enabling further manipulation or analysis of these components.
    - Inputs:
        - `ipadapter_pipe` (Required): The `ipadapter_pipe` parameter represents the bundled pipeline from which individual components are to be extracted. It is crucial for enabling the decomposition process. Type should be `IPADAPTER_PIPE`.
    - Outputs:
        - `ipadapter`: Extracts the IP adapter component from the bundled pipeline. Type should be `IPADAPTER`.
        - `model`: Retrieves the model component from the bundled pipeline. Type should be `MODEL`.
        - `clip_vision`: Extracts the CLIP vision component, if present, from the bundled pipeline. Type should be `CLIP_VISION`.
        - `insight_face`: Retrieves the InsightFace component, if applicable, from the bundled pipeline. Type should be `INSIGHTFACE`.
