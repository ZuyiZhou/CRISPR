"""
Image-C3 dataset

Loads teacher labels and prepares training data.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ImageC3Dataset(Dataset):
    """
    Image-C3 training dataset

    Supports two loading formats:

    Format 1 (QA format):
    {
        "id": "0_0",
        "image_path": "...",
        "question": "...",
        "teacher_answer": "...",
        "original_answer": "...",
        "source": "..."
    }

    Format 2 (Caption format):
    {
        "id": 0,
        "image_paths": ["..."],
        "caption": "...",
        "source": "..."
    }
    """

    def __init__(
        self,
        data_path: str,
        encoder_processor,
        decoder_processor,
        max_length: int = 512,
        max_samples: Optional[int] = None,
    ):
        self.data_path = data_path
        self.encoder_processor = encoder_processor
        self.decoder_processor = decoder_processor
        self.max_length = max_length

        # Load data
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    # Check and convert format
                    normalized = self._normalize_sample(sample)
                    if normalized:
                        self.samples.append(normalized)

        print(f"Loaded {len(self.samples)} samples from {data_path}")

    def _normalize_sample(self, sample: Dict) -> Optional[Dict]:
        """Convert samples of different formats into a unified format"""
        # Format 1: QA format (already has image_path, question, teacher_answer)
        if all(k in sample for k in ["image_path", "question", "teacher_answer"]):
            return {
                "id": sample["id"],
                "image_path": sample["image_path"],
                "question": sample["question"],
                "answer": sample["teacher_answer"],
            }

        # Format 2: Caption format (image_paths, caption)
        if "image_paths" in sample and "caption" in sample:
            image_paths = sample["image_paths"]
            if isinstance(image_paths, list) and len(image_paths) > 0:
                return {
                    "id": sample["id"],
                    "image_path": image_paths[0],  # use the first image
                    "question": "Describe this image in detail.",
                    "answer": sample["caption"],
                }

        # Unsupported format
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Load image
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            # Return a gray image as fallback
            image = Image.new("RGB", (384, 384), (128, 128, 128))

        return {
            "id": sample["id"],
            "image": image,
            "question": sample["question"],
            "answer": sample["answer"],
        }


class ImageC3Collator:
    """
    Data collator

    Processes batch data and produces model inputs.
    """

    def __init__(
        self,
        encoder_processor,
        decoder_processor,
        max_length: int = 512,
        prepare_teacher_inputs: bool = True,
    ):
        self.encoder_processor = encoder_processor
        self.decoder_processor = decoder_processor
        self.max_length = max_length
        self.prepare_teacher_inputs = prepare_teacher_inputs

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of data

        Returns:
        - encoder_inputs: used to extract patch tokens
        - decoder_inputs: used to generate the answer
        - labels: training labels
        """
        images = [item["image"] for item in batch]
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]

        # === Encoder inputs (used to extract patch tokens) ===
        # Build the message format
        encoder_messages_list = []
        for image, question in zip(images, questions):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            encoder_messages_list.append(messages)

        # Process encoder inputs (mainly to obtain the image's pixel_values)
        encoder_texts = []
        for msgs in encoder_messages_list:
            text = self.encoder_processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            encoder_texts.append(text)

        encoder_inputs = self.encoder_processor(
            text=encoder_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # === Teacher inputs (with image tokens, used for distillation) ===
        # Build the message format including images, with the answer included so teacher logits can be computed
        teacher_messages_list = []
        for image, question, answer in zip(images, questions, answers):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                },
                {
                    "role": "assistant",
                    "content": answer,
                }
            ]
            teacher_messages_list.append(messages)

        # Process teacher inputs
        teacher_texts = []
        for msgs in teacher_messages_list:
            text = self.decoder_processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            teacher_texts.append(text)

        # Do not truncate teacher inputs, to keep image tokens intact
        teacher_inputs = self.decoder_processor(
            text=teacher_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False,  # do not truncate; keep the full set of image tokens
        )

        # === Decoder/Student inputs (text format, compressed tokens replace the image) ===
        # Build the question part
        decoder_prompts = []
        for question in questions:
            # Simple format: Question: xxx\nAnswer:
            prompt = f"Question: {question}\nAnswer:"
            decoder_prompts.append(prompt)

        # Build the full input (question + answer)
        decoder_full_texts = []
        for prompt, answer in zip(decoder_prompts, answers):
            full_text = prompt + " " + answer
            decoder_full_texts.append(full_text)

        # Tokenize the decoder inputs
        decoder_inputs = self.decoder_processor.tokenizer(
            decoder_full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Build labels (loss is only computed on the answer part)
        labels = decoder_inputs["input_ids"].clone()

        # Find, for each sample, where the text after "Answer:" begins
        # Simplified handling: mask out the preceding portion (the prompt part)
        prompt_inputs = self.decoder_processor.tokenizer(
            decoder_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # For each sample, mask out the prompt part
        for i in range(labels.shape[0]):
            prompt_len = prompt_inputs["attention_mask"][i].sum().item()
            labels[i, :prompt_len] = -100  # ignore prompt

        # Also mask out padding
        labels[decoder_inputs["attention_mask"] == 0] = -100

        # === Teacher labels (loss is computed on the answer part) ===
        teacher_labels = teacher_inputs["input_ids"].clone()
        # Need to find where the assistant reply begins
        # Simplified: use the prompt part (without the answer) to locate it
        teacher_prompt_messages = []
        for image, question in zip(images, questions):
            msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            teacher_prompt_messages.append(msgs)

        teacher_prompt_texts = []
        for msgs in teacher_prompt_messages:
            text = self.decoder_processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            teacher_prompt_texts.append(text)

        teacher_prompt_inputs = self.decoder_processor(
            text=teacher_prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False,  # do not truncate
        )

        # Mask out the prompt part
        for i in range(teacher_labels.shape[0]):
            prompt_len = teacher_prompt_inputs["attention_mask"][i].sum().item()
            teacher_labels[i, :prompt_len] = -100

        # Mask out padding
        teacher_labels[teacher_inputs["attention_mask"] == 0] = -100

        return {
            # Encoder inputs (for extracting patch tokens)
            "pixel_values": encoder_inputs.get("pixel_values"),
            "image_grid_thw": encoder_inputs.get("image_grid_thw"),

            # Teacher inputs (full image + text, used for distillation)
            "teacher_input_ids": teacher_inputs["input_ids"],
            "teacher_attention_mask": teacher_inputs["attention_mask"],
            "teacher_pixel_values": teacher_inputs.get("pixel_values"),
            "teacher_image_grid_thw": teacher_inputs.get("image_grid_thw"),
            "teacher_labels": teacher_labels,

            # Student/Decoder inputs (compressed tokens + text)
            "input_ids": decoder_inputs["input_ids"],
            "attention_mask": decoder_inputs["attention_mask"],
            "labels": labels,

            # Metadata
            "ids": [item["id"] for item in batch],
        }


def create_dataloader(
    data_path: str,
    encoder_processor,
    decoder_processor,
    batch_size: int = 4,
    max_length: int = 512,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create a data loader"""

    dataset = ImageC3Dataset(
        data_path=data_path,
        encoder_processor=encoder_processor,
        decoder_processor=decoder_processor,
        max_length=max_length,
        max_samples=max_samples,
    )

    collator = ImageC3Collator(
        encoder_processor=encoder_processor,
        decoder_processor=decoder_processor,
        max_length=max_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    return dataloader
