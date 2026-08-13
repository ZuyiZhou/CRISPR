"""
Image-C3 dataset loader

Supports Caption and VQA tasks
"""

import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Any
from pathlib import Path


class ImageC3Dataset(Dataset):
    """
    Image-C3 training dataset

    Data format (JSONL):
    {
        "id": "0",
        "image_path": "data/food101_1k/images/000000.jpg",
        "task": "caption",
        "texts": [{"user": "...", "assistant": "..."}],
        "source": "food101"
    }
    """

    def __init__(
        self,
        data_path: str,
        processor,
        max_length: int = 512,
        image_size: int = 384,
        task: str = "caption",  # "caption" or "vqa"
    ):
        """
        Args:
            data_path: Path to the JSONL file
            processor: Qwen-VL processor
            max_length: Maximum text length
            image_size: Image size
            task: Task type
        """
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size
        self.task = task
        self.data_dir = Path(data_path).parent

        # Load data
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Load image
        image_path = sample["image_path"]
        if not os.path.isabs(image_path):
            # Relative path, resolved against the project root
            image_path = str(self.data_dir.parent.parent / image_path)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            image = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))

        # Get text
        texts = sample.get("texts", [])
        if texts:
            user_text = texts[0].get("user", "Describe this image.")
            assistant_text = texts[0].get("assistant", "")
        else:
            user_text = "Describe this image."
            assistant_text = ""

        return {
            "id": sample.get("id", str(idx)),
            "image": image,
            "user_text": user_text,
            "assistant_text": assistant_text,
            "task": sample.get("task", self.task),
        }


def collate_fn_qwen_vl(batch: List[Dict], processor, max_length: int = 512):
    """
    Collate function for the Qwen-VL format

    Converts a batch into the model input format
    """
    images = []
    conversations = []

    for item in batch:
        images.append(item["image"])

        # Build the conversation format
        conv = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["image"]},
                    {"type": "text", "text": item["user_text"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": item["assistant_text"]},
                ],
            },
        ]
        conversations.append(conv)

    # Process with the processor
    # The Qwen-VL processor requires special handling
    texts = []
    for conv in conversations:
        text = processor.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

    # Process images and text
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Create labels (used for loss computation)
    # labels equal input_ids, but the prompt portion is set to -100
    labels = inputs["input_ids"].clone()

    # Find where the assistant response starts; everything before that is set to -100
    # Simplified handling: assume the first half is the prompt and the second half is the response
    # A more precise approach would need to set this based on actual token positions

    # Here we let the entire sequence contribute to the loss (simplified version)
    # This can later be optimized to only compute loss on the assistant portion

    inputs["labels"] = labels

    return inputs


class ImageC3DataLoader:
    """
    Wrapper around the Image-C3 data loader
    """

    def __init__(
        self,
        data_path: str,
        processor,
        batch_size: int = 4,
        max_length: int = 512,
        num_workers: int = 4,
        shuffle: bool = True,
    ):
        self.dataset = ImageC3Dataset(
            data_path=data_path,
            processor=processor,
            max_length=max_length,
        )

        self.processor = processor
        self.max_length = max_length

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda batch: collate_fn_qwen_vl(batch, processor, max_length),
            pin_memory=True,
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


# ─── Simplified Dataset (for quick testing) ───

class SimpleImageCaptionDataset(Dataset):
    """
    Simplified dataset: returns the image and text directly, without extra processing
    Used for quickly validating the pipeline
    """

    def __init__(self, data_path: str, transform=None):
        self.data_dir = Path(data_path).parent
        self.transform = transform

        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        print(f"SimpleDataset: Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image_path = sample["image_path"]
        if not os.path.isabs(image_path):
            image_path = str(self.data_dir.parent.parent / image_path)

        try:
            image = Image.open(image_path).convert("RGB")
        except:
            image = Image.new("RGB", (384, 384), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        # Get the caption
        texts = sample.get("texts", [])
        caption = texts[0].get("assistant", "") if texts else ""
        prompt = texts[0].get("user", "Describe this image.") if texts else "Describe this image."

        return {
            "image": image,
            "prompt": prompt,
            "caption": caption,
            "id": sample.get("id", str(idx)),
        }
