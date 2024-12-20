- `Create Grid Image`: This node is designed to create a grid image from a collection of image files. It allows for customization of the grid layout, including the number of columns, cell size, and border properties. The node supports filtering images by file extension and can include images from subfolders based on the user's choice.
    - Inputs:
        - `images_path` (Required): The path to the directory containing the images to be included in the grid. The node checks for the existence of this path and filters images based on allowed extensions. Type should be `STRING`.
        - `pattern_glob` (Required): A glob pattern to filter the images in the specified path. It allows for more granular control over which images are included based on their filenames. Type should be `STRING`.
        - `include_subfolders` (Required): A flag indicating whether to include images from subdirectories of the specified path. This allows for recursive image inclusion based on the user's preference. Type should be `COMBO[STRING]`.
        - `border_width` (Required): The width of the border around each image in the grid. A width of 0 or less means no border is added. Type should be `INT`.
        - `number_of_columns` (Required): Specifies the number of columns in the grid layout. This affects the overall arrangement and appearance of the grid image. Type should be `INT`.
        - `max_cell_size` (Required): The maximum size for each cell in the grid, controlling the dimensions of the images in the grid. Type should be `INT`.
        - `border_red` (Required): The red component of the border color, allowing for customization of the border's appearance. Type should be `INT`.
        - `border_green` (Required): The green component of the border color, contributing to the customization of the border's appearance. Type should be `INT`.
        - `border_blue` (Required): The blue component of the border color, further allowing for customization of the border's appearance. Type should be `INT`.
    - Outputs:
        - `image`: The generated grid image as a tensor, suitable for further processing or visualization. Type should be `IMAGE`.
