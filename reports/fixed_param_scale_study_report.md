# fixed_param_scale_study Results

## Configuration Comparison

| Configuration   | Total Parameters   | Training Time (s)   | Final Accuracy (%)   |
|:----------------|:-------------------|:--------------------|:---------------------|
| 1x_resnet34     | 21.90M             | 175.2 ± 1.5         | 88.05 ± 0.51         |
| 2x_resnet18     | 22.98M             | 194.6 ± 2.2         | 85.76 ± 1.63         |
| 3x_resnet9      | 7.45M              | 409.9 ± 0.9         | 93.80 ± 0.48         |



## Efficiency Metrics

| Configuration   |   Accuracy/Parameter (×10⁶) |   Accuracy/Second |
|:----------------|----------------------------:|------------------:|
| 1x_resnet34     |                       4.02  |             0.503 |
| 2x_resnet18     |                       3.732 |             0.441 |
| 3x_resnet9      |                      12.596 |             0.229 |



## Model Architecture Details

```

```
