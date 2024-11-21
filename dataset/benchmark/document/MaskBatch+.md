- `MaskBatch+`: The MaskBatch+ node is designed for batch processing of mask images, specifically for combining two mask images into a single batch. It ensures compatibility between mask sizes through resizing operations if necessary, facilitating the integration of masks from different sources or dimensions into a unified batch format.
    - Inputs:
        - `mask1` (Required): The first mask image to be combined into the batch. Its dimensions are checked against the second mask to ensure compatibility. Type should be `MASK`.
        - `mask2` (Required): The second mask image to be combined with the first. If its dimensions differ from the first mask, it is resized to match, ensuring uniformity in the batch. Type should be `MASK`.
    - Outputs:
        - `mask`: The combined batch of the two input masks, returned as a single tensor. This facilitates further processing or analysis of the masks as a unified entity. Type should be `MASK`.