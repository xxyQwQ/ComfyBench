- `Paste By Mask`: The Paste By Mask node enables the pasting of one image onto another using a specified mask to define the paste location. It supports various resizing behaviors to ensure the pasted image fits the mask area appropriately, and optionally allows for mask mapping to control the pasting process more granularly.
    - Inputs:
        - `image_base` (Required): The base image onto which another image will be pasted. It serves as the background for the operation. Type should be `IMAGE`.
        - `image_to_paste` (Required): The image to be pasted onto the base image. This image is manipulated according to the mask and resize behavior to fit appropriately. Type should be `IMAGE`.
        - `mask` (Required): A mask that defines where on the base image the pasting should occur. The mask's shape and values determine the pasting area. Type should be `IMAGE`.
        - `resize_behavior` (Required): Specifies how the image to paste should be resized to fit the mask area, with options like keeping the ratio, filling the area, or matching the source size. Type should be `COMBO[STRING]`.
        - `mask_mapping_optional` (Optional): Optionally used to control which part of the pasting image goes onto which part of the base image, based on a mask mapping obtained from a separate operation. Type should be `MASK_MAPPING`.
    - Outputs:
        - `image`: The result of pasting an image onto another with the specified mask, producing a new image that combines elements of both according to the defined parameters. Type should be `IMAGE`.