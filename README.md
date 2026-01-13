# Pantropical landuse Model Documentation

This document describes the pantropical land use model used for monitoring land use following deforestation using remote sensing.

## Model Overview

The model is an Attention U-Net with fusion mechanisms, specifically designed for land use monitoring using multi-source satellite data (Sentinel-1 and Sentinel-2) along with geographical information.

### Model Architecture

- **Base Architecture**: Attention U-Net with multi-input fusion
- **Original Framework**: TensorFlow 2.10.0

## Installation & Dependencies for local users

Create and activate the virtual environment and install the package as follows:

```
mamba create -n tf214_py39 python=3.9 tensorflow=2.14.0 onnx tf2onnx ipykernel -c conda-forge -y && mamba activate tf214_py39 && python -m ipykernel install --user --name=tf214_py39 --display-name="TF 2.14 + ONNX"
```
Then install these packages as well:
 ```
mamba install earthengine-api geemap rasterio numpy matplotlib ipywidgets onnxruntime requests folium pyproj tqdm -q
```

## Interactive notebook

Run the analysis interactively in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/<USERNAME>/<REPO>/blob/main/Land_Use_Following_Deforestation_MonitorV1.ipynb
)
Open this notebook colab https://colab.research.google.com/drive/1B6x3lQJWMu3iwS3iYbIFA1OHx8BQ2DpP#scrollTo=OHzOEaWRps0z and follow instractions:

```
#The notebook allows you to:

🖼️ Draw or upload a Region of Interest (ROI) on an interactive map

🧠 Automatically selects AI model based on location (Africa, Southeast Asia, Latin America)

🛰️ Downloads and preprocesses Sentinel-1 + Sentinel-2 + elevation + indices

🌾 Predicts land use categories over deforested areas only using ONNX models

🗺️ Side-by-side map of RGB imagery + follow-up land use prediction

📤 Export predictions as GeoTIFF for GIS analysis
```

## Model Input Specifications

The model expects a single input tensor with the following specifications:

- **Input Name**: "input"
- **Shape**: `[1, 64, 64, 17]`
  - Batch size: 1 (fixed)
  - Height: 64 pixels
  - Width: 64 pixels
  - Channels: 17 (combined features)
- **Data Type**: float32 (elem_type: 1)

### Input Channel Organization
The 17 input channels are organized as follows:

1. **Sentinel-2 Bands** (Channels 0-8):
   - Blue, Green, Red
   - Red Edge 1, 2, 3
   - NIR
   - SWIR 1, 2
   - *Note: These bands are normalized using log-transformation and percentile-based scaling*

2. **Radar Data** (Channels 9-10):
   - VV polarization (normalized to [-25, 0] range)
   - VH polarization (normalized to [-30, -5] range)

3. **Geographical Information** (Channels 11-13):
   - Altitude (normalized to [-400, 8000] range)
   - Longitude (normalized to [-180, 180] range)
   - Latitude (normalized to [-60, 60] range)

4. **Additional Features** (Channel 14):
   - Derived indices (NDVI, EVI, NDRI)???

## Output Specifications

- **Output Name**: "activation_65"
- **Shape**: `[1, 64, 64, 22]`
  - Batch size: 1 (fixed)
  - Height: 64 pixels
  - Width: 64 pixels
  - Classes: 1 (probability distribution over crop type)
- **Data Type**: float32 (elem_type: 1)

### Crop Classes
The model predicts 25, 22, and 23 land use types for Africa, Latin America, and Southeast Asia, respectively. Each pixel in the output contains a probability distribution over this class.

Example format for each pixel:
```python
classes = [
    # Add crop classes here
    "class_0",  # e.g., "Oil Palm"
    "class_1",  # e.g., "Other"
]
```

## Model Conversion Details

The model was converted from TensorFlow to ONNX using `tf2onnx` with the following specifications:
- Original model: TensorFlow SavedModel/HDF5 format
- Target ONNX opset: 13
- Conversion tool: tf2onnx version 1.16.1

### Conversion Validation
- Maximum numerical difference between TensorFlow and ONNX outputs: 5.66e-06
- Validation performed using random input data
- All operations successfully mapped to ONNX operators


## Usage Example

```python
import onnxruntime as ort
import numpy as np

# Load the ONNX model
session = ort.InferenceSession("comcrop_udf_test.onnx")

# Prepare input data (example)
input_data = np.random.rand(1, 64, 64, 17).astype(np.float32)

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
predictions = session.run([output_name], {input_name: input_data})[0]

# Get predicted class
predicted_classes = np.argmax(predictions, axis=-1)
```

## Model Performance Notes

1. The model preserves the attention mechanisms from the original TensorFlow implementation
2. Includes upsampling operations with deprecated 'tf_half_pixel_for_nn' attribute (opset 13 warning)
3. Maintains the multi-scale feature fusion capabilities of the original model

## Deployment Considerations

1. The model requires normalized input data according to the specified ranges for each channel
2. Memory requirements: Approximately 150MB for model weights
3. Compatible with any ONNX Runtime supported platform
4. Recommended to use ONNX Runtime version 1.20.0 or higher


---

## ONNX Model Description

ONNX Model Description generated by [`onnx.helper.printable_graph(onnx_model.graph)`];

The model processes the 15 input channels by splitting them into three groups, each handled by a distinct pathway:

1. **Channels 0–11 (12 channels):** Processed by a U-Net-like structure for feature extraction and segmentation.
2. **Channels 12–13 (2 channels):** Handled by a second U-Net-like structure, mirroring the first.
3. **Channels 14–16 (3 channels):** Fed into a dense (fully connected) network to generate features, likely used as attention maps to guide the U-Nets.

These pathways are fused together, with attention mechanisms emphasizing key regions, to produce a final segmentation map. This design is ideal for tasks where input channels represent diverse information, such as medical imaging or satellite imagery.

---

## Components

The model relies on several standard deep learning components for image processing:

- **Convolutional Layers:** Extract spatial features using filters (e.g., 3x3 kernels), typically followed by batch normalization and ReLU activation for stability and efficiency.
- **Max Pooling:** Reduces feature map sizes (e.g., 64x64 to 32x32) during the encoder phase, capturing essential patterns while lowering computational load.
- **Transpose Convolutions:** Upsample feature maps in the decoder phase, restoring resolution (e.g., 16x16 to 32x32).
- **Attention Mechanisms:** Weight features based on attention maps from the dense network, focusing on critical areas.
- **Concatenation:** Merges features from different network parts, such as skip connections in U-Nets or the final fusion step.
- **Sigmoid:** Outputs class probabilities for the final segmentation map.

---

## Detailed Structure

Here’s a step-by-step walkthrough of how the model processes the input.

### 1. Input Splitting
The input tensor `[1, 64, 64, 17]` is divided along the channel dimension into 15 tensors of shape `[1, 64, 64, 1]`, then grouped as:
- **Channels 0–11:** Concatenated into `[1, 64, 64, 12]`.
- **Channels 12–13:** Concatenated into `[1, 64, 64, 2]`.
- **Channels 14–16:** Concatenated into `[1, 64, 64, 3]`.

Each group follows a unique processing path.

### 2. Dense Network (Channels 14–16)
- **Input:** `[1, 64, 64, 3]`, transposed to `[1, 3, 64, 64]`.
- **Structure:** Mirrors the U-Net for channels 10–11:
  - Encoder: Downsamples to `[1, 512, 8, 8]` through levels with 64, 128, 256, and 512 filters.
  - Decoder: Upsamples back to `[1, 64, 64, 64]` with attention and skip connections.
- **Output:** `[1, 64, 64, 64]` feature map.
- **Purpose:** Provides an attention signal or context for the U-Nets.

### 3. U-Net for Channels 12–13
- **Input:** `[1, 64, 64, 2]`, transposed to `[1, 2, 64, 64]`.
- **Encoder (Downsampling):**
  - **Level 1:** Two 3x3 conv layers (64 filters), output `[1, 64, 64, 64]`, then max pooling to `[1, 64, 32, 32]`.
  - **Level 2:** Two 3x3 conv layers (128 filters), output `[1, 128, 32, 32]`, then max pooling to `[1, 128, 16, 16]`.
  - **Level 3:** Two 3x3 conv layers (256 filters), output `[1, 256, 16, 16]`, then max pooling to `[1, 256, 8, 8]`.
  - **Bottom:** Two 3x3 conv layers (512 filters), output `[1, 512, 8, 8]`.
- **Decoder (Upsampling with Attention):**
  - **Level 3:** Upsampled to `[1, 256, 16, 16]`, combined with encoder features via skip connection and attention, processed with conv layers.
  - **Level 2:** Upsampled to `[1, 128, 32, 32]`, attention applied, concatenated, and conv layers.
  - **Level 1:** Upsampled to `[1, 64, 64, 64]`, attention applied, concatenated, and conv layers.
- **Output:** `[1, 64, 64, 64]` feature map.

### 4. U-Net for Channels 0–11
- **Input:** `[1, 64, 64, 10]`, transposed to `[1, 10, 64, 64]`.
- **Structure:** Mirrors the U-Net for channels 10–11:
  - Encoder: Downsamples to `[1, 512, 8, 8]` through levels with 64, 128, 256, and 512 filters.
  - Decoder: Upsamples back to `[1, 64, 64, 64]` with attention and skip connections.
- **Output:** `[1, 64, 64, 64]` feature map.

### 5. Fusion and Output
- **Concatenation:** Outputs from both U-Nets and the dense network (resized to `[1, 256, 64, 64]`) are combined into `[1, 384, 64, 64]` (64 + 64 + 256 = 384, though channel counts may vary).
- **Final Convolution:** A 1x1 conv layer with 22 filters produces `[1, 22, 64, 64]`.
- **Softmax:** Generates class probabilities.
- **Output:** Transposed to `[1, 64, 64, 1]`, a segmentation map with 1 class per pixel.

---

## Workflow Summary

1. **Splitting:** The 17-channel input is divided into three parts for specialized processing.
2. **Feature Extraction:**
   - U-Nets extract hierarchical features from channels 0–11 and 12–13 using encoder-decoder paths with skip connections.
   - The dense network processes channels 12–14 into a guiding feature map.
3. **Attention:** The dense network’s output weights U-Net features during upsampling, highlighting key areas.
4. **Fusion:** All features are combined to integrate multi-channel information.
5. **Segmentation:** A final layer classifies each pixel into one 1 class.

---

## Design Philosophy

This architecture is tailored for:
- **Multi-Channel Inputs:** Handles diverse data types across 15 channels.
- **Attention-Driven Focus:** Improves accuracy by emphasizing important regions.
- **Detailed Segmentation:** Preserves spatial details via U-Net skip connections, perfect for tasks like medical or land-use segmentation.

---

## Summary

The `Attention_UNet_Fusion` model combines three U-Nets
network to process a 17-channel, 64x64 input. Channels 0–11, 12–13 and 14 - 16
re handled by U-Nets for feature extraction, while channels 12–14 guide 
attention via a dense network. The fused output becomes a 64x64 segmentation map with 1 class, making it a tool for binary image segmentation tasks.
