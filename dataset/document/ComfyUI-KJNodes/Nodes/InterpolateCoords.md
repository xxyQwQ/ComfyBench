# Interpolate Coords
## Documentation
- Class name: `InterpolateCoords`
- Category: `KJNodes/experimental`
- Output node: `False`

The InterpolateCoords node is designed for interpolating a set of coordinates based on a specified curve. It processes input coordinates and an interpolation curve to generate a new set of coordinates that follow the given curve, effectively transforming the spatial layout of the original points.
## Input types
### Required
- **`coordinates`**
    - A string representing the original coordinates to be interpolated. This parameter is crucial for defining the starting point of the interpolation process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`interpolation_curve`**
    - A list of floating-point numbers representing the curve along which the coordinates will be interpolated. This parameter determines the shape and trajectory of the interpolation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `List[float]`
## Output types
- **`coordinates`**
    - Comfy dtype: `STRING`
    - A string representation of the interpolated coordinates, formatted as a list of dictionaries with 'x' and 'y' keys.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class InterpolateCoords:
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("coordinates",)
    FUNCTION = "interpolate"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = """
Interpolates coordinates based on a curve.   
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "coordinates": ("STRING", {"forceInput": True}),
                "interpolation_curve": ("FLOAT", {"forceInput": True}),
                
        },
    } 

    def interpolate(self, coordinates, interpolation_curve):
        # Parse the JSON string to get the list of coordinates
        coordinates = json.loads(coordinates.replace("'", '"'))

        # Convert the list of dictionaries to a list of (x, y) tuples for easier processing
        coordinates = [(coord['x'], coord['y']) for coord in coordinates]

        # Calculate the total length of the original path
        path_length = sum(np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[i-1])) 
                        for i in range(1, len(coordinates)))

        # Initialize variables for interpolation
        interpolated_coords = []
        current_length = 0
        current_index = 0

        # Iterate over the normalized curve
        for normalized_length in interpolation_curve:
            target_length = normalized_length * path_length # Convert to the original scale
            while current_index < len(coordinates) - 1:
                segment_start, segment_end = np.array(coordinates[current_index]), np.array(coordinates[current_index + 1])
                segment_length = np.linalg.norm(segment_end - segment_start)
                if current_length + segment_length >= target_length:
                    break
                current_length += segment_length
                current_index += 1

            # Interpolate between the last two points
            if current_index < len(coordinates) - 1:
                p1, p2 = np.array(coordinates[current_index]), np.array(coordinates[current_index + 1])
                segment_length = np.linalg.norm(p2 - p1)
                if segment_length > 0:
                    t = (target_length - current_length) / segment_length
                    interpolated_point = p1 + t * (p2 - p1)
                    interpolated_coords.append(interpolated_point.tolist())
                else:
                    interpolated_coords.append(p1.tolist())
            else:
                # If the target_length is at or beyond the end of the path, add the last coordinate
                interpolated_coords.append(coordinates[-1])

        # Convert back to string format if necessary
        interpolated_coords_str = "[" + ", ".join([f"{{'x': {round(coord[0])}, 'y': {round(coord[1])}}}" for coord in interpolated_coords]) + "]"
        print(interpolated_coords_str)

        return (interpolated_coords_str,)

```
