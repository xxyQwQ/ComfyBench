- `Cut By Mask`: The 'Cut By Mask' node is designed to precisely extract or isolate parts of an image based on a specified mask. It allows for the resizing of the cut-out image to specific dimensions, optionally handling multiple image segments in a batch when provided with mask mappings from a 'Separate Mask Components' node.
    - Inputs:
        - `image` (Required): The input image to be processed. It serves as the primary canvas from which parts will be cut out based on the mask. Type should be `IMAGE`.
        - `mask` (Required): The mask that defines the areas of the input image to be cut out. It acts as a stencil, specifying which parts of the image are to be extracted or isolated. Type should be `IMAGE`.
        - `force_resize_width` (Required): An optional width to which the cut-out image(s) will be resized. If specified, it overrides the natural size of the cut-out segments, enabling uniformity in output dimensions. Type should be `INT`.
        - `force_resize_height` (Required): An optional height to which the cut-out image(s) will be resized. Similar to force_resize_width, it allows for the standardization of output sizes across different cut-outs. Type should be `INT`.
        - `mask_mapping_optional` (Optional): An optional mapping of mask components, typically provided by a 'Separate Mask Components' node. It enables the cutting of multiple image segments in a single operation, based on distinct mask parts. Type should be `MASK_MAPPING`.
    - Outputs:
        - `image`: The output image(s) after being cut by the mask. This can be a single image or multiple images, depending on whether mask mapping was used, resized as specified. Type should be `IMAGE`.