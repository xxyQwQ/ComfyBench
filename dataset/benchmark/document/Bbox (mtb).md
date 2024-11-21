- `Bbox (mtb)`: The Bbox (mtb) node is designed to generate a bounding box (bbox) based on specified dimensions. It abstracts the process of defining a rectangular area within an image or a frame, facilitating operations like cropping or region of interest (ROI) identification.
    - Inputs:
        - `x` (Required): Specifies the x-coordinate of the top-left corner of the bounding box, serving as a starting point for the rectangle's horizontal dimension. Type should be `INT`.
        - `y` (Required): Specifies the y-coordinate of the top-left corner of the bounding box, serving as a starting point for the rectangle's vertical dimension. Type should be `INT`.
        - `width` (Required): Defines the width of the bounding box, determining how far it extends horizontally from the x-coordinate. Type should be `INT`.
        - `height` (Required): Defines the height of the bounding box, determining how far it extends vertically from the y-coordinate. Type should be `INT`.
    - Outputs:
        - `bbox`: The output is a tuple representing the bounding box, structured as (x, y, width, height), which can be used for further image processing tasks. Type should be `BBOX`.