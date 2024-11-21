- `CR Comic Panel Templates`: This node is designed to generate comic panel layouts based on predefined templates. It dynamically creates and arranges panels within a page, allowing for customization of panel sizes, colors, and border styles to fit various comic storytelling needs.
    - Inputs:
        - `page_width` (Required): Specifies the width of the page on which the comic panels will be drawn, influencing the layout's scale and the size of individual panels. Type should be `INT`.
        - `page_height` (Required): Determines the height of the page, affecting the overall layout and the vertical size of the comic panels. Type should be `INT`.
        - `template` (Required): Defines the template used for the comic panel layout, which can be a predefined template or a custom layout specified by the user. Type should be `COMBO[STRING]`.
        - `reading_direction` (Required): Indicates the direction in which the comic should be read, affecting the arrangement and flow of the panels. Type should be `COMBO[STRING]`.
        - `border_thickness` (Required): Defines the thickness of the borders around each panel, contributing to the visual style of the comic. Type should be `INT`.
        - `outline_thickness` (Required): Sets the thickness of the outline around the page and panels, impacting the comic's aesthetic appeal. Type should be `INT`.
        - `outline_color` (Required): Specifies the color of the outline around the panels and page, adding to the visual design of the comic. Type should be `COMBO[STRING]`.
        - `panel_color` (Required): Determines the color of the panel backgrounds, influencing the mood and style of the comic. Type should be `COMBO[STRING]`.
        - `background_color` (Required): Sets the background color of the page, framing the panels and contributing to the overall visual theme. Type should be `COMBO[STRING]`.
        - `images` (Optional): A collection of images to be placed within the comic panels. This parameter is crucial for populating the panels with visual content, significantly enhancing the storytelling aspect of the comic. Type should be `IMAGE`.
        - `custom_panel_layout` (Optional): Allows for the specification of a custom layout for the comic panels, offering flexibility in design beyond predefined templates. Type should be `STRING`.
        - `outline_color_hex` (Optional): Provides a hexadecimal color code for the outline color, offering precise color customization. Type should be `STRING`.
        - `panel_color_hex` (Optional): Gives a hexadecimal color code for the panel color, allowing for exact color selection. Type should be `STRING`.
        - `bg_color_hex` (Optional): Specifies a hexadecimal color code for the background color, enabling detailed color customization. Type should be `STRING`.
    - Outputs:
        - `image`: The final image output containing the generated comic panel layout. Type should be `IMAGE`.
        - `show_help`: A string output providing guidance or additional information related to the generated comic panel layout. Type should be `STRING`.