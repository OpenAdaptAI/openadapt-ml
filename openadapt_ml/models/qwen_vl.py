from __future__ import annotations

from typing import Any, Dict, List, Optional

from PIL import Image
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

from openadapt_ml.models.base_adapter import BaseVLMAdapter, get_default_device


def _process_vision_info(messages: List[Dict[str, Any]]) -> tuple[list[list[Any]], list[list[Any]]]:
    """Minimal stand-in for qwen_vl_utils.process_vision_info.

    For our use case we only need to extract image/video entries from the
    message content structure expected by Qwen-VL examples, where each
    message has a `content` list of dicts with `type` in {"image", "video"}.

    Returns (image_inputs, video_inputs), each a list-of-lists suitable for
    passing to AutoProcessor.
    """

    image_inputs: list[list[Any]] = []
    video_inputs: list[list[Any]] = []

    current_images: list[Any] = []
    current_videos: list[Any] = []

    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t == "image":
                current_images.append(item.get("image"))
            elif t == "video":
                current_videos.append(item.get("video"))

    if current_images:
        image_inputs.append(current_images)
    if current_videos:
        video_inputs.append(current_videos)

    return image_inputs, video_inputs


class QwenVLAdapter(BaseVLMAdapter):
    """Adapter for Qwen-family VLMs using Hugging Face + PEFT.

    This is a minimal skeleton that:
    - loads a base model + processor
    - optionally applies a LoRA adapter
    - implements the BaseVLMAdapter interface

    The exact chat/image encoding and loss masking will be filled in
    once we wire a concrete training loop.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any,
        device: Optional[torch.device] = None,
        version: str = "qwen3",
    ) -> None:
        super().__init__(model=model, processor=processor, device=device)
        self.version = version

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        lora_config: Optional[LoraConfig | Dict[str, Any]] = None,
        load_in_4bit: bool = False,
        device: Optional[torch.device] = None,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
    ) -> "QwenVLAdapter":
        """Load base Qwen model + processor and attach optional LoRA adapter.

        Args:
            max_pixels: Maximum image size in pixels (e.g., 512*512=262144 for faster training).
                        If None, uses model default (very large).
            min_pixels: Minimum image size in pixels. If None, uses model default.
        """

        if "Qwen3-VL" in model_name or "Qwen3VL" in model_name:
            version = "qwen3"
            model_cls = Qwen3VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_name or "Qwen2_5" in model_name:
            version = "qwen2_5"
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unrecognized Qwen-VL model name: {model_name}")

        processor = AutoProcessor.from_pretrained(model_name)

        # Configure image resolution for faster training
        if max_pixels is not None and hasattr(processor, 'image_processor'):
            processor.image_processor.max_pixels = max_pixels
            print(f"Set max_pixels to {max_pixels} ({int(max_pixels**0.5)}x{int(max_pixels**0.5)} approx)")
        if min_pixels is not None and hasattr(processor, 'image_processor'):
            processor.image_processor.min_pixels = min_pixels

        model_kwargs: Dict[str, Any] = {}
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        model = model_cls.from_pretrained(model_name, **model_kwargs)

        # Support two modes for LoRA:
        # - config-based (no weights_path): create a fresh adapter.
        # - weights-based (weights_path in dict): load an existing adapter.
        lora_weights_path: Optional[str] = None
        lora_cfg_clean: Optional[LoraConfig | Dict[str, Any]] = None

        if lora_config is not None:
            if isinstance(lora_config, dict):
                lora_weights_path = lora_config.get("weights_path")
                lora_cfg_clean = {k: v for k, v in lora_config.items() if k != "weights_path"}
            else:
                lora_cfg_clean = lora_config

        if lora_weights_path:
            # Load an existing adapter onto the base model.
            model = PeftModel.from_pretrained(model, lora_weights_path)
        elif lora_cfg_clean is not None:
            if isinstance(lora_cfg_clean, dict):
                lora_cfg_clean = LoraConfig(**lora_cfg_clean)
            model = get_peft_model(model, lora_cfg_clean)  # type: ignore[arg-type]

        if device is None:
            device = get_default_device()

        return cls(model=model, processor=processor, device=device, version=version)

    def prepare_inputs(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:  # type: ignore[override]
        """Convert SFT samples into model inputs for Qwen-VL.

        Supports true batching by processing multiple samples simultaneously.
        Uses processor.apply_chat_template with padding=True and truncation=True
        for multi-sample tokenization. Computes assistant-only labels per sample.

        Args:
            batch: List of SFT-style samples, each with "images" and "messages" keys.

        Returns:
            Dict with input_ids, attention_mask, pixel_values, image_grid_thw, and labels.
        """

        if self.version == "qwen3":
            # Build batch of messages for all samples
            batch_messages_full: List[List[Dict[str, Any]]] = []
            batch_messages_user_only: List[List[Dict[str, Any]]] = []

            for sample in batch:
                image_paths = sample["images"]
                if not image_paths:
                    raise ValueError("Sample is missing image paths")
                image_path = image_paths[0]

                messages = sample["messages"]
                user_text = ""
                assistant_text = ""
                for m in messages:
                    role = m.get("role")
                    if role == "user":
                        user_text = m.get("content", "")
                    elif role == "assistant":
                        assistant_text = m.get("content", "")

                # Full messages (user + assistant)
                qwen_messages_full: List[Dict[str, Any]] = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": user_text},
                        ],
                    },
                ]
                if assistant_text:
                    qwen_messages_full.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": assistant_text}],
                    })
                batch_messages_full.append(qwen_messages_full)

                # User-only messages (for label masking)
                qwen_messages_user_only: List[Dict[str, Any]] = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": user_text},
                        ],
                    }
                ]
                batch_messages_user_only.append(qwen_messages_user_only)

            # Tokenize full batch with padding and truncation
            inputs_full = self.processor.apply_chat_template(  # type: ignore[call-arg]
                batch_messages_full,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            # Tokenize user-only batch for label masking
            inputs_user = self.processor.apply_chat_template(  # type: ignore[call-arg]
                batch_messages_user_only,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            input_ids_full = inputs_full["input_ids"]  # [batch_size, seq_len]
            input_ids_user = inputs_user["input_ids"]  # [batch_size, seq_len_user]

            # Initialize labels with full input_ids, then mask per sample
            labels = input_ids_full.clone()

            # Compute assistant-only labels per sample
            batch_size = input_ids_full.size(0)
            for i in range(batch_size):
                sample = batch[i]
                messages = sample["messages"]
                assistant_text = ""
                for m in messages:
                    if m.get("role") == "assistant":
                        assistant_text = m.get("content", "")

                if assistant_text:
                    # Find where user prompt ends and assistant response begins
                    # The user-only sequence should be a prefix of the full sequence
                    full_ids = input_ids_full[i]
                    user_ids = input_ids_user[i]

                    # Remove padding from user_ids to find actual sequence length
                    # Padding token is typically 0 or a special value
                    # For Qwen, we look for the first occurrence of pad token
                    pad_token_id = self.processor.tokenizer.pad_token_id
                    user_ids_no_pad = user_ids[user_ids != pad_token_id] if pad_token_id is not None else user_ids
                    user_len = len(user_ids_no_pad)

                    # Check if user sequence is a prefix of full sequence
                    if user_len <= full_ids.size(0):
                        # Mask everything except assistant tokens
                        labels[i, :] = -100
                        # Only supervise on tokens after the user prompt
                        labels[i, user_len:] = full_ids[user_len:]

            # Ensure padding tokens are masked in labels
            if hasattr(self.processor.tokenizer, 'pad_token_id') and self.processor.tokenizer.pad_token_id is not None:
                labels[input_ids_full == self.processor.tokenizer.pad_token_id] = -100

            inputs_full["labels"] = labels
            return inputs_full

        else:  # qwen2_5
            # Build batch of messages
            batch_messages: List[List[Dict[str, Any]]] = []
            batch_texts: List[str] = []
            all_image_inputs: List[List[Any]] = []
            all_video_inputs: List[List[Any]] = []

            for sample in batch:
                image_paths = sample["images"]
                if not image_paths:
                    raise ValueError("Sample is missing image paths")
                image_path = image_paths[0]

                messages = sample["messages"]
                user_text = ""
                assistant_text = ""
                for m in messages:
                    role = m.get("role")
                    if role == "user":
                        user_text = m.get("content", "")
                    elif role == "assistant":
                        assistant_text = m.get("content", "")

                qwen_messages: List[Dict[str, Any]] = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": user_text},
                        ],
                    }
                ]
                if assistant_text:
                    qwen_messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": assistant_text}],
                    })

                batch_messages.append(qwen_messages)

                # Convert to text for this sample
                text = self.processor.apply_chat_template(  # type: ignore[call-arg]
                    qwen_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                batch_texts.append(text)

                # Extract vision info
                image_inputs, video_inputs = _process_vision_info(qwen_messages)
                if image_inputs:
                    all_image_inputs.extend(image_inputs)
                if video_inputs:
                    all_video_inputs.extend(video_inputs)

            # Process batch with padding
            videos_arg = all_video_inputs if all_video_inputs else None
            inputs = self.processor(  # type: ignore[call-arg]
                text=batch_texts,
                images=all_image_inputs if all_image_inputs else None,
                videos=videos_arg,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            # For qwen2_5, use full sequence supervision for now
            # (can be refined to assistant-only if needed)
            input_ids = inputs["input_ids"]
            labels = input_ids.clone()

            # Mask padding tokens
            if hasattr(self.processor.tokenizer, 'pad_token_id') and self.processor.tokenizer.pad_token_id is not None:
                labels[input_ids == self.processor.tokenizer.pad_token_id] = -100

            inputs["labels"] = labels
            return inputs

    def compute_loss(self, inputs: Dict[str, Any]) -> torch.Tensor:  # type: ignore[override]
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # Hugging Face causal LM models return `loss` when `labels` are provided.
        return outputs.loss  # type: ignore[no-any-return]

    def generate(self, sample: Dict[str, Any], max_new_tokens: int = 64) -> str:  # type: ignore[override]
        """Generate assistant text for a single SFT-style sample.

        We pass system + user messages to the chat template with
        `add_generation_prompt=True` and let the model generate the
        assistant continuation.
        """

        image_paths = sample["images"]
        if not image_paths:
            raise ValueError("Sample is missing image paths")
        image_path = image_paths[0]

        messages = sample["messages"]
        user_text = ""
        for m in messages:
            if m.get("role") == "user":
                user_text = m.get("content", "")

        qwen_messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        if self.version == "qwen3":
            inputs = self.processor.apply_chat_template(  # type: ignore[call-arg]
                qwen_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
        else:
            text = self.processor.apply_chat_template(  # type: ignore[call-arg]
                qwen_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = _process_vision_info(qwen_messages)
            videos_arg = video_inputs if video_inputs else None
            inputs = self.processor(  # type: ignore[call-arg]
                text=[text],
                images=image_inputs,
                videos=videos_arg,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

        with torch.no_grad():
            generation = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Decode only the GENERATED tokens (not the input prompt)
        # This is critical - otherwise we return "user Goal: ... assistant ..." instead of just the response
        input_len = inputs["input_ids"].shape[1]
        generated_ids = generation[:, input_len:]
        text = self.processor.batch_decode(  # type: ignore[call-arg]
            generated_ids,
            skip_special_tokens=True,
        )[0]
        return text

    def save_checkpoint(self, path: str) -> None:
        """Save the LoRA adapter weights to a directory."""
        from pathlib import Path
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        # Save the PEFT adapter (LoRA weights only, not base model)
        self.model.save_pretrained(str(save_path))
