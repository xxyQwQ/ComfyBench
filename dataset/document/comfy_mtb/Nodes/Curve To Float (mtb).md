# Curve To Float (mtb)
## Documentation
- Class name: `Curve To Float (mtb)`
- Category: `mtb/curve`
- Output node: `False`

This node specializes in transforming a FLOAT_CURVE into a FLOAT or FLOATS. It abstractly focuses on the conversion process, leveraging the curve's structure to produce either a singular value or a sequence of values, tailored to the interpolation requirements.
## Input types
### Required
- **`curve`**
    - Represents a curve defined by a series of points, each with X and Y coordinates, serving as the primary data structure for generating FLOAT or FLOATS output through interpolation.
    - Comfy dtype: `FLOAT_CURVE`
    - Python dtype: `Dict[str, Dict[str, float]]`
- **`steps`**
    - Specifies the number of intervals for interpolating Y values along the curve, affecting the granularity of the FLOATS output or the precision of the FLOAT output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`floats`**
    - Comfy dtype: `FLOATS`
    - A list of interpolated FLOAT values derived from the input FLOAT_CURVE, based on the specified number of steps.
    - Python dtype: `List[float]`
- **`float`**
    - Comfy dtype: `FLOAT`
    - A single interpolated FLOAT value from the input FLOAT_CURVE, suitable for scenarios requiring a singular value.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MTB_CurveToFloat:
    """Convert a FLOAT_CURVE to a FLOAT or FLOATS"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "curve": ("FLOAT_CURVE", {"forceInput": True}),
                "steps": ("INT", {"default": 10, "min": 2}),
            },
        }

    RETURN_TYPES = ("FLOATS", "FLOAT")
    FUNCTION = "do_curve"

    CATEGORY = "mtb/curve"

    def do_curve(self, curve, steps):
        log.debug(f"Curve: {curve}")

        # sort by x (should be handled by the widget)
        sorted_points = sorted(curve.items(), key=lambda item: item[1]["x"])
        # Extract X and Y values
        x_values = [point[1]["x"] for point in sorted_points]
        y_values = [point[1]["y"] for point in sorted_points]
        # Calculate step size
        step_size = (max(x_values) - min(x_values)) / (steps - 1)

        # Interpolate Y values for each step
        interpolated_y_values = []
        for step in range(steps):
            current_x = min(x_values) + step_size * step

            # Find the indices of the two points between which the current_x falls
            idx1 = max(idx for idx, x in enumerate(x_values) if x <= current_x)
            idx2 = min(idx for idx, x in enumerate(x_values) if x >= current_x)

            # If the current_x matches one of the points, no interpolation is needed
            if current_x == x_values[idx1]:
                interpolated_y_values.append(y_values[idx1])
            elif current_x == x_values[idx2]:
                interpolated_y_values.append(y_values[idx2])
            else:
                # Interpolate Y value using linear interpolation
                y1 = y_values[idx1]
                y2 = y_values[idx2]
                x1 = x_values[idx1]
                x2 = x_values[idx2]
                interpolated_y = y1 + (y2 - y1) * (current_x - x1) / (x2 - x1)
                interpolated_y_values.append(interpolated_y)

        return (interpolated_y_values, interpolated_y_values)

```
