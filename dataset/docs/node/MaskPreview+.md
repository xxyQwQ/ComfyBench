- `MaskPreview+`: The MaskPreview node is designed for generating a preview of masks by reshaping and expanding the input mask tensor, and then saving these images to a temporary directory with optional compression. It facilitates the visualization of mask data by converting it into a more interpretable form.
    - Parameters:
    - Inputs:
        - `mask`: The mask input is the primary data that this node processes, transforming it into a visual format for preview purposes. Type should be `MASK`.
    - Outputs: