import argparse
import io
import os
import random
from collections import Counter

import cv2
import evaluate
import kagglehub
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image
from scipy.fftpack import dct
from torchvision import transforms as T
from transformers import (
   Trainer,
   TrainingArguments,
   ViTForImageClassification,
   ViTImageProcessor,
)

torch.cuda.set_per_process_memory_fraction(0.85)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "google/vit-base-patch16-224"
NUM_LABELS = 2
ID2LABEL = {0: "FAKE", 1: "REAL"}
LABEL2ID = {"FAKE": 0, "REAL": 1}
SEED = 42


def set_seed(seed: int = SEED) -> None:
   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 1. Forensic Analysis Tools
# ---------------------------------------------------------------------------

def get_dct_score(image: Image.Image) -> float:
   img_gray = np.array(image.convert("L")).astype(np.float32)
   dct_2d = dct(dct(img_gray.T, norm='ortho').T, norm='ortho')
   high_freq_energy = np.sum(np.abs(dct_2d[-32:, -32:]))
   return float(high_freq_energy)


def get_noise_residual(image_path):
   img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
   if img is None:
       raise FileNotFoundError(f"Could not load image at {image_path}")

   kernel = np.array([[-1, -1, -1],
                      [-1, 8, -1],
                      [-1, -1, -1]])
   residual = cv2.filter2D(img, -1, kernel)
   residual_vis = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)
   return img, residual_vis


# SRM high-pass kernel
SRM_KERNEL = torch.tensor([[-1., -1., -1.],
                            [-1.,  8., -1.],
                            [-1., -1., -1.]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1) 

def apply_srm_to_tensor(pixel_values: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        residual = F.conv2d(pixel_values, SRM_KERNEL, groups=3, padding=1)
        residual = torch.clamp(residual, -3.0, 3.0) / 3.0 
    return residual


def apply_srm_filter(image: Image.Image) -> Image.Image:
   img_gray = np.array(image.convert("L"))
   kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
   residual = cv2.filter2D(img_gray, -1, kernel)
   residual_vis = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)
   return Image.fromarray(cv2.cvtColor(residual_vis, cv2.COLOR_GRAY2RGB))


def generate_paper_figure(real_path, ai_path):
   orig_fp, noise_fp = get_noise_residual(real_path)
   orig_ai, noise_ai = get_noise_residual(ai_path)

   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   axes[0, 0].imshow(orig_fp, cmap='gray')
   axes[0, 0].set_title("Authentic Image (Grayscale)")
   axes[0, 1].imshow(noise_fp, cmap='magma')
   axes[0, 1].set_title("Authentic Noise Residual (Stochastic)")

   axes[1, 0].imshow(orig_ai, cmap='gray')
   axes[1, 0].set_title("AI-Generated Image (Grayscale)")
   axes[1, 1].imshow(noise_ai, cmap='magma')
   axes[1, 1].set_title("AI Noise Residual (Structural/Periodic)")

   plt.tight_layout()
   plt.savefig("paper_noise_analysis.png")
   print("Success: Figure saved as 'paper_noise_analysis.png'")
   plt.show()


# ---------------------------------------------------------------------------
# 2. Training Infrastructure
# ---------------------------------------------------------------------------

def download_dataset() -> str:
   return kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")


def get_datasets(data_root: str) -> dict:
   train_dir = os.path.join(data_root, "train")
   test_dir = os.path.join(data_root, "test")
   train_ds = load_dataset("imagefolder", data_dir=train_dir, split="train")
   eval_ds = load_dataset("imagefolder", data_dir=test_dir, split="train")
   return {"train": train_ds, "eval": eval_ds}


def build_processor() -> ViTImageProcessor:
   return ViTImageProcessor.from_pretrained(MODEL_NAME)


# ---------------------------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------------------------

def jpeg_compress(img: Image.Image, quality_range=(30, 95)) -> Image.Image:
    """Simulate JPEG compression artifacts to improve robustness."""
    buf = io.BytesIO()
    q = random.randint(*quality_range)
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


TRAIN_AUGMENT = T.Compose([
    T.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    T.Lambda(lambda img: jpeg_compress(img) if random.random() < 0.3 else img),
    T.RandomGrayscale(p=0.05),
])


def preprocess_datasets(ds: dict, processor: ViTImageProcessor) -> dict:
    def _make_transform(augment: bool):
        def _transform(batch):
            images = [img.convert("RGB") for img in batch["image"]]
            if augment:
                images = [TRAIN_AUGMENT(img) for img in images]
            encoded = processor(images=images, return_tensors="pt")
            encoded["labels"] = torch.tensor(batch["label"])
            return encoded
        return _transform

    for split_name in ds:
        augment = split_name == "train"
        tag = "augmented" if augment else "standard"
        print(f"Setting up {tag} transform for {split_name} split...")
        ds[split_name].set_transform(_make_transform(augment))

    return ds


FREEZE_LAYERS = 8  # freeze first N of 12 encoder layers


def build_model(model_path_or_name: str = MODEL_NAME, freeze: bool = False) -> ViTForImageClassification:
   model = ViTForImageClassification.from_pretrained(
       model_path_or_name, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
       ignore_mismatched_sizes=True, output_attentions=True
   )
   if freeze:
       # Freeze embeddings
       for param in model.vit.embeddings.parameters():
           param.requires_grad = False
       # Freeze first N encoder layers
       for layer in model.vit.encoder.layer[:FREEZE_LAYERS]:
           for param in layer.parameters():
               param.requires_grad = False
       trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
       total = sum(p.numel() for p in model.parameters())
       print(f"Frozen first {FREEZE_LAYERS} layers: {trainable:,} / {total:,} params trainable ({100*trainable/total:.1f}%)")
   return model


def compute_metrics_fn():
   accuracy_metric = evaluate.load("accuracy")
   f1_metric = evaluate.load("f1")

   def compute_metrics(eval_pred):
       logits, labels = eval_pred
       predictions = np.argmax(logits, axis=-1)
       acc = accuracy_metric.compute(predictions=predictions, references=labels)
       f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")
       return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

   return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
   return logits[0] if isinstance(logits, tuple) else logits


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([item["pixel_values"].squeeze(0) for item in batch]),
        "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    }


# ---------------------------------------------------------------------------
# 3. Execution Loops
# ---------------------------------------------------------------------------

def train_and_evaluate(output_dir: str, do_train: bool, do_eval: bool):
   set_seed(SEED)
   data_root = download_dataset()
   ds = get_datasets(data_root)
   processor = build_processor()
   ds = preprocess_datasets(ds, processor)
   model = build_model(freeze=True)

   training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    label_smoothing_factor=0.1,
    per_device_train_batch_size=64,     
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,      
    num_train_epochs=5,
    eval_strategy="epoch",              
    save_strategy="epoch",
    save_total_limit=2,                
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    bf16=True,                          
    gradient_checkpointing=True,       
    dataloader_num_workers=6,           
    dataloader_prefetch_factor=4,       
    remove_unused_columns=False,
    report_to="none"                   
   )

   trainer = Trainer(
       model=model, 
       args=training_args, 
       train_dataset=ds["train"],
       eval_dataset=ds["eval"],         
       compute_metrics=compute_metrics_fn(),
       data_collator=collate_fn, 
       preprocess_logits_for_metrics=preprocess_logits_for_metrics
   )

   if do_train:
       trainer.train()
       trainer.save_model(output_dir)
       processor.save_pretrained(output_dir)
   if do_eval:
       print("\nEvaluation results:", trainer.evaluate())


def predict_with_explainability(image_path: str, model_dir: str):
   try:
       processor = ViTImageProcessor.from_pretrained(model_dir)
   except:
       processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

   model = build_model(model_dir)
   device = "cuda" if torch.cuda.is_available() else "cpu"
   model.to(device)
   model.eval()

   raw_image = Image.open(image_path).convert("RGB")
   
   inputs = processor(images=raw_image, return_tensors="pt")
   inputs = {k: v.to(device) for k, v in inputs.items()}

   dct_score = get_dct_score(raw_image)
   _, noise_res = get_noise_residual(image_path)

   with torch.no_grad():
       outputs = model(**inputs)

   attentions = outputs.attentions[-1]
   mask = attentions[0].mean(dim=0)[0, 1:].reshape(14, 14).detach().cpu().numpy()
   mask_resized = cv2.resize(mask, (raw_image.size[0], raw_image.size[1]))

   probs = F.softmax(outputs.logits, dim=-1)
   conf, pred_id = probs.max(dim=-1)
   label = ID2LABEL[pred_id.item()]

   plt.figure(figsize=(18, 6))
   plt.subplot(1, 3, 1)
   plt.title(f"Prediction ({label})\nConf: {conf.item():.2%}")
   plt.imshow(raw_image)
   plt.axis("off")

   plt.subplot(1, 3, 2)
   plt.title(f"Attention Heatmap\nDCT: {dct_score:.2f}")
   plt.imshow(raw_image)
   plt.imshow(mask_resized, cmap='jet', alpha=0.5)
   plt.axis("off")

   plt.subplot(1, 3, 3)
   plt.title("Noise Residual\n(SRM-style)")
   plt.imshow(noise_res, cmap='magma')
   plt.axis("off")

   plt.tight_layout()
   plt.savefig("prediction_output.png", dpi=300, bbox_inches='tight')
   plt.show()
   return label, conf.item(), dct_score


# ---------------------------------------------------------------------------
# 4. Main Entry
# ---------------------------------------------------------------------------

def main():
   parser = argparse.ArgumentParser(description="ViT Universal AI Detector")
   parser.add_argument("--output_dir", type=str, default="./vit-cifake-output")
   parser.add_argument("--do_train", action="store_true")
   parser.add_argument("--do_eval", action="store_true")
   parser.add_argument("--predict_image", type=str, help="Single image inference")
   parser.add_argument("--compare_real", type=str, help="Real image for paper figure")
   parser.add_argument("--compare_ai", type=str, help="AI image for paper figure")
   args = parser.parse_args()

   if args.compare_real and args.compare_ai:
       generate_paper_figure(args.compare_real, args.compare_ai)
   elif args.predict_image:
       predict_with_explainability(args.predict_image, args.output_dir)
   elif args.do_train or args.do_eval:
       train_and_evaluate(args.output_dir, args.do_train, args.do_eval)
   else:
       parser.print_help()


if __name__ == "__main__":
   main()