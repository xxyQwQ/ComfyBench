- `Conditioning Grid (cond)`: This node is designed to apply conditioning to a grid layout, where each cell within the grid can be individually conditioned based on text inputs. It automates the process of encoding text inputs using a CLIP model and then applying these encoded conditionings to specific areas within a grid, facilitating the generation of complex, grid-based visual layouts with varied content.
    - Inputs:
        - `columns` (Required): Specifies the number of columns in the grid, determining the grid's horizontal layout and how many cells it contains. Type should be `INT`.
        - `rows` (Required): Specifies the number of rows in the grid, determining the grid's vertical layout and how many cells it contains. Type should be `INT`.
        - `width` (Required): The width of each cell in the grid, in pixels. This affects the resolution and aspect ratio of the content within each cell. Type should be `INT`.
        - `height` (Required): The height of each cell in the grid, in pixels. This affects the resolution and aspect ratio of the content within each cell. Type should be `INT`.
        - `strength` (Required): Controls the intensity of the applied conditioning, allowing for fine-tuning of how prominently the text inputs influence the generated content. Type should be `FLOAT`.
        - `base` (Required): The base text input for the grid's overall theme or background. This input sets the foundational conditioning layer upon which additional cell-specific conditionings are applied. Type should be `CONDITIONING`.
    - Outputs:
        - `conditioning`: The resulting conditioning for the grid, ready to be used for generating content within each cell based on the provided text inputs. Type should be `CONDITIONING`.