- `ImpactSEGSToMaskList`: This node is designed to convert segmentation data (SEGS) into a list of masks. It serves as a utility within the Impact Pack, facilitating the transformation of complex segmentation formats into a more universally applicable mask format, thereby enabling further image processing and analysis tasks.
    - Inputs:
        - `segs` (Required): The 'segs' parameter represents the segmentation data to be converted into masks. It is crucial for the node's operation as it provides the raw segmentation information that will be transformed into a list of individual masks. Type should be `SEGS`.
    - Outputs:
        - `mask`: The output is a list of masks derived from the input segmentation data. Each mask corresponds to a segment in the input, allowing for detailed analysis and manipulation of individual segments. Type should be `MASK`.