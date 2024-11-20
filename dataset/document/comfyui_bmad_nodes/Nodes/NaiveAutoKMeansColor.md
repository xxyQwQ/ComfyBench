---
tags:
- Color
---

# NaiveAutoKMeansColor
## Documentation
- Class name: `NaiveAutoKMeansColor`
- Category: `Bmad/CV/Color A.`
- Output node: `False`

This node is designed to automatically determine the optimal number of colors for K-means clustering on an image, and then apply the K-means algorithm to segment the image into these colors. It simplifies the process of color quantization by abstracting away the need for manual specification of the number of clusters, making it easier to use for tasks such as image simplification or color analysis.
## Input types
### Required
- **`image`**
    - The input image on which K-means clustering will be performed. This is the primary data the node operates on to determine the optimal number of colors.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`max_k`**
    - The maximum number of colors (clusters) to consider for determining the optimal number through the algorithm. It sets an upper limit on the number of clusters to evaluate.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`rc_threshold`**
    - A threshold for the relative compactness of clusters, used to help determine the optimal number of clusters by comparing the compactness of a cluster to the first computed compactness.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`max_iterations`**
    - The maximum number of iterations the K-means algorithm will execute, providing a stopping criterion for convergence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`eps`**
    - The epsilon value for convergence criteria, specifying the minimum amount of change required for an iteration to be considered as making progress.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image after applying K-means clustering with the determined optimal number of colors, segmented into these colors.
    - Python dtype: `torch.Tensor`
- **`int`**
    - Comfy dtype: `INT`
    - The optimal number of colors (clusters) determined by the algorithm for the input image.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class NaiveAutoKMeansColor:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "max_k": ("INT", {"default": 8, "min": 3, "max": 16}),

            # besides looking like the elbow,
            #  a k's compactness divided the by first computed compactness should be below this value
            "rc_threshold": ("FLOAT", {"default": .5, "max": 1, "min": 0.01, "step": 0.01}),

            "max_iterations": ("INT", {"default": 100}),
            "eps": ("FLOAT", {"default": .2, "step": 0.05})
        }}

    RETURN_TYPES = ("IMAGE", "INT")
    FUNCTION = "get_colors"
    CATEGORY = f"{cv_category_path}/Color A."

    def get_colors(self, image, max_k, rc_threshold, max_iterations, eps):
        image = tensor2opencv(image, 3)
        pixels = image.reshape(-1, 3)
        pixels = np.float32(pixels)

        def normalize(vector):
            return vector / np.linalg.norm(vector)

        def compute_angle_at_k(prev_k_c, k_c, next_k_c):
            p_km1 = np.array([-1, prev_k_c, 0])
            p_k = np.array([0, k_c, 0])
            p_kp1 = np.array([1, next_k_c, 0])
            v1 = normalize(p_km1 - p_k)
            v2 = normalize(p_kp1 - p_k)
            return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

        # define criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, max_iterations, eps)

        # compute k means and check for the elbow
        #  here the elbow is the edgiest point on the compactness graph
        best_angle = 7  # max is pi, when the line is perfectly straight; and the objective is to minimize the angle
        max_c = best_rc = best_k = None
        current_k_data = prev_k_data = best_k_data = None
        for k in range(1, max_k + 2):
            next_k_data = cv.kmeans(pixels, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

            if max_c is None:
                max_c = next_k_data[0]

            if next_k_data[0] == 0:
                # if it is a perfect fit, leave the method
                # avoids unneeded computation, and division by zero on k = 1
                best_k_data = next_k_data
                best_k = k
                break

            if k > 2:
                rc = current_k_data[0] / max_c
                angle = compute_angle_at_k(prev_k_data[0] / max_c, rc, next_k_data[0] / max_c)
                if angle < best_angle or best_rc > rc_threshold:
                    best_angle = angle
                    best_k_data = current_k_data
                    best_rc = rc
                    best_k = k - 1

            prev_k_data = current_k_data
            current_k_data = next_k_data

        # convert back into uint8, and make original image
        center = np.uint8(best_k_data[2])
        res = center[best_k_data[1].flatten()]
        res2 = res.reshape((image.shape))
        res2 = opencv2tensor(res2)
        return (res2, best_k)

```
