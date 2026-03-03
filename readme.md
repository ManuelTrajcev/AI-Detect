# Forensic Noise-Guided Vision Transformer for AI-Generated Image Detection

A forensic detection pipeline that combines **SRM (Spatial Rich Model) noise residual extraction** with a fine-tuned **Vision Transformer (ViT)** to distinguish authentic photographs from AI-generated images.

## Key Results

| Model | Accuracy | F1 Score | Eval Loss |
|-------|----------|----------|-----------|
| **SRM (Noise Residuals)** | **96.79%** | **96.80%** | 0.0970 |
| RGB (Raw Pixels) | 50.68% | 47.46% | 0.7317 |

The SRM-trained model outperforms the RGB baseline by **46 percentage points** — the RGB model performs at chance level, confirming that explicit forensic preprocessing is essential.

## How It Works

1. **SRM Filtering** — A high-pass Laplacian kernel extracts the noise residual from each image, stripping semantic content and exposing the noise floor that encodes image provenance.
2. **ViT Classification** — A fine-tuned `vit-base-patch16-224` classifies the noise residuals as FAKE or REAL.
3. **Explainability** — A 3-panel visualization shows the prediction, attention heatmap, and noise residual for every inference.

## Project Structure

```
train_noise.py          # SRM model — training, evaluation, and inference
train_rgb.py            # RGB baseline — training, evaluation, and inference
index.html              # Research paper with full comparative analysis
vit-cifake-output/      # Trained SRM model weights
vit-cifake-output2/     # Trained RGB model weights
test_images/            # Test images for inference
predictions/            # Saved prediction visualizations
```

## Usage

**Train the SRM model:**
```bash
python train_noise.py --do_train --output_dir ./vit-cifake-output
```

**Train the RGB baseline:**
```bash
python train_rgb.py --do_train --output_dir ./vit-cifake-output2
```

**Run inference with explainability visualization:**
```bash
python train_noise.py --predict_image test_images/example.jpg --output_dir ./vit-cifake-output
```

**Generate noise comparison figure:**
```bash
python train_noise.py --compare_real path/to/real.jpg --compare_ai path/to/fake.jpg
```

## Requirements

- Python 3.10+
- PyTorch with CUDA
- `transformers`, `datasets`, `evaluate`, `kagglehub`
- `opencv-python`, `scipy`, `matplotlib`, `Pillow`

## Dataset

Trained on [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) (100,000 images, balanced REAL/FAKE split). Downloaded automatically via `kagglehub` at training time.
