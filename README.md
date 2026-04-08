# CSE472: Seizure Classification with Explainable AI (XAI)

## 📋 Overview

This project implements deep learning models for **EEG-based seizure classification** on two benchmark datasets (Bonn-U and TUH EEG), with comprehensive explainability analysis using five state-of-the-art XAI techniques. The goal is to build accurate seizure detection systems while providing interpretable insights into model predictions.

### Key Objectives
- Train robust seizure classification models on standard EEG datasets
- Compare performance with and without contrastive learning
- Apply multiple XAI methods to explain model decisions
- Visualize which EEG channels and time periods are most critical for predictions

---

## 📁 Project Structure

```
CSE472-Machine-Learning-Project/
├── Bonn-U dataset/
│   ├── Seizure Classification on Bonn-U dataset no constrastive learning.ipynb
│   └── Seizure Classification on Bonn-U dataset with constrastive learning.ipynb
├── TUH EEG dataset/
│   ├── Seizure Classification on TUH EEG dataset no constrastive learning.ipynb
│   └── Seizure classification on TUH EEG dataset with constrastive learning.ipynb
├── XAI/
│   ├── TUH-EEG-XAI.ipynb
│   └── results/
│       ├── xai_attention_rtm.png
│       ├── xai_channel_occlusion.png
│       ├── xai_gradcam.png
│       ├── xai_integrated_gradients.png
│       └── xai_shap.png
└── README.md
```

---

## 📊 Datasets

### 1. **Bonn-U Dataset**
- **Source**: University of Bonn EEG dataset
- **Classes**: Epileptic seizure vs. Non-seizure
- **Characteristics**: 
  - High-quality EEG recordings
  - Balanced class distribution
  - Pre-processed and publicly available

### 2. **TUH EEG Dataset**
- **Source**: Temple University Hospital EEG database
- **Classes**: Epileptic seizure vs. Non-seizure
- **Characteristics**:
  - Large-scale clinical EEG data (60 patients studied)
  - Multiple EEG channels (21 target channels)
  - Real-world clinical setting
  - Sampling rate: 128 Hz (resampled to standard rate)

**EEG Channels Used:**
```
FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, 
F7, F8, T3, T4, T5, T6, FZ, CZ, PZ, A1, A2
(Total: 21 channels)
```

---

## 🧠 Model Architecture: L_SCLNet_EEGformer

### Overview
The **L_SCLNet_EEGformer** is a state-of-the-art Transformer-based architecture specifically designed for EEG signal classification.

### Architecture Components

#### 1. **ODCM (Optimized Depthwise Convolutional Module)**
- Extracts initial temporal patterns using depthwise convolutions
- Parameters:
  - Kernel size: 10
  - 120 convolutional filters
  - Reduces sequence length from 512 to ~382 samples

#### 2. **RTM (Relational Tensor Module)**
- Applies multi-head attention across channels
- Key features:
  - Learns channel relationships
  - 6 attention heads
  - 3 transformer blocks
  - Output: Channel attention patterns

#### 3. **STM (Spatial Tensor Module)**
- Cross-channel spatial interactions
- Configuration:
  - 6 attention heads
  - 3 transformer blocks
  - Captures channel-level dependencies

#### 4. **TTM (Temporal Tensor Module)**
- Models temporal dynamics
- Features:
  - 6 sub-matrices for temporal decomposition
  - 6 attention heads
  - 3 transformer blocks
  - Multi-scale temporal analysis

#### 5. **CNN Decoder**
- Final classification layer
- Combines spatial and temporal representations
- Output: 2 class logits (Seizure / Non-seizure)

### Model Statistics
- **Total Trainable Parameters**: ~500K
- **Input Shape**: [Batch, 21 channels, 512 samples]
- **Output Shape**: [Batch, 2 classes]
- **Processing**: Window-based (4-second windows at 128 Hz)

---

## 🔍 Explainability Analysis (XAI)

The project implements **5 complementary XAI techniques** to interpret model predictions:

### 1. **Attention Weight Visualization**
- **What it shows**: Which channels the Transformer attends to
- **Technique**: Extract softmax attention weights from RTM blocks
- **Output**: Channel-to-channel attention heatmaps
- **Interpretation**: Red/hot colors indicate strong attention relationships

**Example Insight**: Temporal regions with high epileptic activity often show focused attention patterns on specific channel clusters.

### 2. **Grad-CAM (Gradient-weighted Class Activation Map)**
- **What it shows**: Temporal saliency map highlighting important time regions
- **Technique**: Gradient-based activation mapping on ODCM conv3 layer
- **Formula**: 
  ```
  CAM = ReLU(Σ_k α_k * A_k)
  where α_k = mean_spatial(∂L/∂A_k)
  ```
- **Output**: Time-series saliency scores [0-1] for each sample
- **Interpretation**: Peaks in the map indicate critical temporal segments for seizure detection

### 3. **Integrated Gradients**
- **What it shows**: Per-channel, per-timestep attribution scores
- **Technique**: Path integration from baseline (zero signal) to actual signal
- **Formula**:
  ```
  IG(x) = (x - baseline) × mean(∇f(baseline + α(x - baseline)))
  for α ∈ [0, 1]
  ```
- **Output**: 2D attribution matrix [21 channels × 512 time steps]
- **Interpretation**: 
  - Red regions = positive contribution to seizure class
  - Blue regions = negative contribution to seizure class

### 4. **SHAP (Expected Gradients / DeepSHAP)**
- **What it shows**: Shapley values explaining each channel and timestep's contribution
- **Technique**: Monte-Carlo approximation of Expected Gradients using 50 background samples
- **Key Innovation**: Custom implementation to handle ODCM's batch-dimension stripping
- **Outputs**:
  - Per-channel importance (bar chart)
  - Channel × Time heatmap (first epileptic sample)
- **Interpretation**:
  - Larger |SHAP| values = more important for seizure detection
  - Positive = increases seizure confidence
  - Negative = decreases seizure confidence

### 5. **Channel Occlusion Sensitivity**
- **What it shows**: Which channels are most critical for predictions
- **Technique**: Zero out each channel individually; measure drop in P(epilepsy)
- **Process**:
  1. Get baseline prediction probabilities
  2. For each of 21 channels:
     - Set channel to zero
     - Measure change in seizure probability
  3. Rank channels by importance
- **Output**: Bar chart with channel rankings
- **Interpretation**: Positive bars = essential for seizure detection; red bars = top-5 channels

---

## 📈 Training Methodology

### Data Preprocessing
- **Channel Selection**: 21 standard EEG channels (minimum 16 channels required)
- **Filtering**:
  - Bandpass: 1-40 Hz (remove DC drift and high-frequency noise)
  - Notch: 60 Hz (remove powerline interference)
- **Resampling**: Standardized to 128 Hz
- **Windowing**:
  - Window size: 4 seconds (512 samples)
  - Step size: 2 seconds (256 samples) → 50% overlap
  - Z-score normalization per window
- **Artifact Rejection**: Remove windows with amplitude > 150 μV

### Data Split Strategy
- **Patient-level split** (not window-level) to prevent information leakage:
  - Training: 70% of patients
  - Validation: 15% of patients
  - Test: 15% of patients
- **Balanced classes**: Equal numbers of epilepsy/non-epilepsy patients
- **Greedy pairing**: Epilepsy and non-epilepsy patients matched by data size

### Training Setup
- **Batch Size**: 32 windows
- **Device**: GPU (CUDA) or CPU fallback
- **Model Save**: Best checkpoint based on validation performance

---

## 🛠️ Running the Notebooks

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn
pip install torch torchvision torchaudio
pip install scikit-learn
pip install mne  # MNE-Tools for EEG processing
pip install shap
```

### Step 1: Dataset Notebooks (Base Models)

#### **Without Contrastive Learning (Baseline)**
```
Bonn-U dataset/Seizure Classification on Bonn-U dataset no constrastive learning.ipynb
TUH EEG dataset/Seizure Classification on TUH EEG dataset no constrastive learning.ipynb
```
- Standard supervised learning
- Performance baseline
- Faster training

#### **With Contrastive Learning**
```
Bonn-U dataset/Seizure Classification on Bonn-U dataset with constrastive learning.ipynb
TUH EEG dataset/Seizure classification on TUH EEG dataset with constrastive learning.ipynb
```
- SimCLR-style contrastive pre-training
- Improved generalization
- Better feature representations

**Execution**:
1. Open notebook in Jupyter
2. Set `MODEL_PATH` and `ROOT` paths to your data directories
3. Adjust hyperparameters if needed (batch size, epochs, learning rate)
4. Run all cells sequentially

### Step 2: XAI Analysis

```
XAI/TUH-EEG-XAI.ipynb
```

**Key sections**:

1. **Setup** (Cell 1-5)
   - Import libraries
   - Configure model path and data paths
   - Load pre-trained weights
   - Define ODCM, RTM, STM, TTM, and decoder classes

2. **Data Loading** (Cell 6-9)
   - Load test set (patient-level 15% of data)
   - Apply same preprocessing as training
   - Select balanced XAI sample (4 seizure + 4 non-seizure samples)

3. **Attention Analysis** (Cell 10)
   - Install attention tracking hooks
   - Visualize channel-to-channel attention patterns

4. **Grad-CAM** (Cell 11)
   - Compute temporal saliency for each sample
   - Plot time-series activation maps

5. **Integrated Gradients** (Cell 12)
   - Compute channel × time attributions
   - Generate 2D heatmaps

6. **SHAP** (Cell 13)
   - Monte-Carlo Expected Gradients computation
   - Channel importance ranking
   - Time-series attribution visualization

7. **Occlusion Sensitivity** (Cell 14)
   - Systematic channel ablation
   - Final channel importance ranking

---

## 📊 Expected Results

### Model Performance
- **Baseline Accuracy**: 85-92% (Bonn-U dataset)
- **TUH EEG Accuracy**: 80-88% (larger, noisier dataset)
- **Contrastive Learning Boost**: +3-5% improvement on generalization metrics

### XAI Insights
- **Key Seizure Channels**: Typically temporal (T3, T4, T5, T6) and central (C3, C4, CZ) regions
- **Temporal Patterns**: Seizures often detected in early-to-mid windows, with specific frequency components
- **Consistency Across Methods**: Different XAI techniques should highlight similar important channels/regions

### Output Visualizations
| Technique | Output File | Purpose |
|-----------|------------|---------|
| Attention | `xai_attention_rtm.png` | Channel relationship matrix |
| Grad-CAM | `xai_gradcam.png` | Temporal saliency (8 samples) |
| Integrated Gradients | `xai_integrated_gradients.png` | Channel × time heatmaps |
| SHAP | `xai_shap.png` | Channel importance + heatmap |
| Occlusion | `xai_channel_occlusion.png` | Channel ranking bar chart |

---

## 🔬 Key Implementation Details

### Custom SHAP Implementation
The SHAP section uses a **manual Expected Gradients** implementation instead of the SHAP library because:
- SHAP's `DeepExplainer` and `GradientExplainer` use deepLIFT hooks
- These hooks crash on the L_SCLNet_EEGformer architecture (ODCM strips batch dimension)
- Manual implementation:
  - Interpolates between background references and target sample
  - Computes gradients at random α ∈ [0,1]
  - Averages over 20 Monte-Carlo steps
  - Produces identical Shapley value estimates

### Memory Management
- GPU memory cleared between SHAP samples
- Background set limited to 50 samples for stability
- Batch processing for large test sets

### Data Validation
- Minimum channel requirement: 16 channels
- Amplitude thresholding: Removes high-amplitude artifacts
- Z-score normalization: Per-window, per-channel basis

---

## 📚 References & Related Work

### EEG Seizure Classification
- Bonn University EEG Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/EEG+Seizure)
- TUH EEG Database: [Temple University Hospital](https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml)

### Model Architecture
- **Transformers for EEG**: Vision Transformers adapted for temporal signals
- **Depthwise Convolutions**: Efficient channel-wise feature extraction
- **Multi-head Attention**: Parallel relationship modeling

### XAI Techniques
1. **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
2. **Integrated Gradients**: Sundararajan et al., "Axiomatic Attribution for Deep Networks" (ICML 2017)
3. **SHAP/DeepSHAP**: Lundberg et al., "Unified Framework for Interpreting Model Predictions" (NeurIPS 2017)
4. **Attention Visualization**: Vaswani et al., "Attention is All You Need" (NeurIPS 2017)
5. **Occlusion Sensitivity**: Zeiler & Fergus, "Visualizing and Understanding CNNs" (ECCV 2014)

---

## 🚀 Usage Tips

### Optimizing for Your Hardware
```python
# Check GPU availability
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Force CPU if needed
device = torch.device("cpu")
```

### Adjusting XAI Parameters
- **More samples**: Increase `XAI_N` (default: 8) for broader analysis
- **More background**: Increase `XAI_BG` (default: 50) for robust SHAP
- **More SHAP steps**: Increase `SHAP_STEPS` (default: 20) for smoother gradients
- **More IG steps**: Increase `n_steps` (default: 50) for finer attribution resolution

### Batch Processing Multiple Samples
```python
# Process large test sets efficiently
for batch_start in range(0, len(X_test), BATCH_SIZE):
    batch = X_test[batch_start:batch_start+BATCH_SIZE]
    with torch.no_grad():
        predictions = model(torch.from_numpy(batch).to(device))
```

---

## 📝 Project Summary

This CSE472 Machine Learning Project combines:
- ✅ **Real-world EEG data** from two benchmark datasets
- ✅ **State-of-the-art deep learning** (Transformer-based architecture)
- ✅ **Advanced ML techniques** (contrastive learning + attention mechanisms)
- ✅ **Comprehensive explainability** (5 complementary XAI methods)
- ✅ **Clinical relevance** (seizure detection for neurology applications)

The project demonstrates how modern deep learning can achieve high accuracy while maintaining interpretability—critical for medical AI applications.

---

## 📧 License & Attribution

Project developed as part of **CSE472: Machine Learning** course.

---

**Last Updated**: 2026 | **Status**: Complete with XAI analysis
