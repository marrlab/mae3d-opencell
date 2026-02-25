"""
Vision-Language Model (VLM) for single-cell microscopy images.

Architecture:
- Vision Encoder: Pretrained MAE 3D encoder
- Projection: MLP to map vision features to LLM embedding space
- Language Model: Mistral-7B / LLaMA-2-7B

Supports:
- 3D microscopy volumes (C, Z, H, W)
- Visual Question Answering
- Two-stage training (projection-only, then fine-tune)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass


@dataclass
class VLMConfig:
    """Configuration for VLM model."""
    # Vision encoder
    vision_encoder_type: str = "mae3d"
    vision_hidden_size: int = 768
    vision_checkpoint: Optional[str] = None
    freeze_vision_encoder: bool = True

    # Projection
    projection_type: str = "mlp"  # "linear" or "mlp"
    num_projection_layers: int = 2
    projection_dropout: float = 0.1

    # Language model
    llm_name: str = "mistralai/Mistral-7B-v0.1"
    llm_hidden_size: int = 4096
    freeze_llm: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Vision tokens
    num_vision_tokens: int = 64  # Number of tokens to represent the image
    use_vision_pooling: bool = True  # Pool vision features before projection

    # Training
    max_length: int = 512


class VisionProjection(nn.Module):
    """
    Projects vision encoder outputs to LLM embedding space.

    Supports:
    - Linear projection
    - MLP projection with multiple layers
    - Learnable query tokens (like Q-Former)
    """

    def __init__(
        self,
        vision_hidden_size: int,
        llm_hidden_size: int,
        num_vision_tokens: int = 64,
        projection_type: str = "mlp",
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vision_hidden_size = vision_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.num_vision_tokens = num_vision_tokens
        self.projection_type = projection_type

        if projection_type == "linear":
            self.projection = nn.Linear(vision_hidden_size, llm_hidden_size)
        elif projection_type == "mlp":
            layers = []
            in_dim = vision_hidden_size
            hidden_dim = (vision_hidden_size + llm_hidden_size) // 2

            for i in range(num_layers - 1):
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                in_dim = hidden_dim

            layers.append(nn.Linear(in_dim, llm_hidden_size))
            self.projection = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")

        # Learnable tokens to pool vision features
        self.vision_queries = nn.Parameter(
            torch.randn(1, num_vision_tokens, vision_hidden_size) * 0.02
        )

        # Cross-attention for pooling
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=vision_hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(vision_hidden_size)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (batch, num_patches, vision_hidden_size)

        Returns:
            projected: (batch, num_vision_tokens, llm_hidden_size)
        """
        batch_size = vision_features.shape[0]

        # Expand queries for batch
        queries = self.vision_queries.expand(batch_size, -1, -1)

        # Cross-attention: queries attend to vision features
        pooled, _ = self.cross_attention(
            query=queries,
            key=vision_features,
            value=vision_features,
        )
        pooled = self.norm(pooled + queries)

        # Project to LLM space
        projected = self.projection(pooled)

        return projected


class CellVLM(nn.Module):
    """
    Vision-Language Model for single-cell microscopy.

    Combines:
    - Pretrained MAE 3D encoder for vision
    - Projection layer
    - LLM for text generation
    """

    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config

        # Build components
        self.vision_encoder = None  # Loaded separately
        self.projection = VisionProjection(
            vision_hidden_size=config.vision_hidden_size,
            llm_hidden_size=config.llm_hidden_size,
            num_vision_tokens=config.num_vision_tokens,
            projection_type=config.projection_type,
            num_layers=config.num_projection_layers,
            dropout=config.projection_dropout,
        )
        self.llm = None  # Loaded separately
        self.tokenizer = None  # Set externally

        # Special tokens
        self.image_token_id = None
        self.image_token = "<image>"

    def load_vision_encoder(self, mae_config, checkpoint_path: Optional[str] = None, model_type: str = "3d"):
        """Load pretrained MAE encoder (2D or 3D)."""
        from lib.networks.mae_vit import MAEViTEncoder, MAEViTDecoder

        # Build MAE model based on type
        if model_type == "2d":
            from lib.models.mae2d import MAE2D
            mae_model = MAE2D(MAEViTEncoder, MAEViTDecoder, mae_config)
        else:
            from lib.models.mae3d import MAE3D
            mae_model = MAE3D(MAEViTEncoder, MAEViTDecoder, mae_config)

        self.model_type = model_type

        # Load checkpoint
        if checkpoint_path:
            print(f"Loading MAE checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)

            # Remove 'module.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v

            mae_model.load_state_dict(new_state_dict, strict=False)

        # Use only encoder
        self.vision_encoder = mae_model.encoder
        self.encoder_pos_embed = mae_model.encoder_pos_embed
        self.patch_size = mae_model.patch_size

        # Freeze if configured
        if self.config.freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.encoder_pos_embed.requires_grad = False
            print("Vision encoder frozen")

    def load_llm(self, device_map: str = "auto"):
        """Load language model with optional LoRA."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading LLM: {self.config.llm_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm_name,
            trust_remote_code=True,
        )

        # Add special tokens
        special_tokens = {"additional_special_tokens": [self.image_token]}
        self.tokenizer.add_special_tokens(special_tokens)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config.llm_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )

        # Resize embeddings for new tokens
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # Apply LoRA if configured
        if self.config.use_lora:
            self._apply_lora()
        elif self.config.freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
            print("LLM frozen")

    def _apply_lora(self):
        """Apply LoRA adapters to LLM."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type=TaskType.CAUSAL_LM,
            )

            self.llm = get_peft_model(self.llm, lora_config)
            print(f"Applied LoRA with r={self.config.lora_r}, alpha={self.config.lora_alpha}")
            self.llm.print_trainable_parameters()

        except ImportError:
            print("PEFT not installed. Run: pip install peft")
            if self.config.freeze_llm:
                for param in self.llm.parameters():
                    param.requires_grad = False

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using vision encoder.

        Args:
            images: (batch, C, Z, H, W) for 3D or (batch, C, H, W) for 2D

        Returns:
            features: (batch, num_patches, hidden_size)
        """
        # Import appropriate patchify function
        if getattr(self, 'model_type', '3d') == '2d':
            from lib.models.mae2d import patchify_image
        else:
            from lib.models.mae3d import patchify_image

        batch_size = images.shape[0]
        device = images.device

        # Patchify
        x = patchify_image(images, self.patch_size)

        # Get positional embeddings
        pos_embed = self.encoder_pos_embed.expand(batch_size, -1, -1).to(device)

        # Forward through encoder
        with torch.set_grad_enabled(not self.config.freeze_vision_encoder):
            features = self.vision_encoder.forward_features(x, pos_embed)

        # Remove CLS token if present
        if features.shape[1] > 1:
            features = features[:, 1:, :]  # Remove CLS token

        return features

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            images: (batch, C, Z, H, W) cell images
            input_ids: (batch, seq_len) tokenized text with <image> placeholder
            attention_mask: (batch, seq_len) attention mask
            labels: (batch, seq_len) labels for language modeling loss

        Returns:
            Dictionary with loss and logits
        """
        batch_size = images.shape[0]
        device = images.device

        # Encode images
        vision_features = self.encode_images(images)

        # Project to LLM space
        image_embeds = self.projection(vision_features)  # (batch, num_vision_tokens, llm_hidden_size)

        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # Find <image> token positions and replace with image embeddings
        image_token_mask = (input_ids == self.image_token_id)

        # Create new embeddings with image tokens inserted
        new_embeds = []
        new_attention_mask = []
        new_labels = [] if labels is not None else None

        for i in range(batch_size):
            # Find position of image token
            img_positions = torch.where(image_token_mask[i])[0]

            if len(img_positions) > 0:
                img_pos = img_positions[0].item()

                # Split text embeddings
                before_img = text_embeds[i, :img_pos]
                after_img = text_embeds[i, img_pos + 1:]

                # Concatenate: before + image_embeds + after
                combined = torch.cat([before_img, image_embeds[i], after_img], dim=0)
                new_embeds.append(combined)

                # Update attention mask
                before_mask = attention_mask[i, :img_pos]
                after_mask = attention_mask[i, img_pos + 1:]
                img_mask = torch.ones(self.config.num_vision_tokens, device=device)
                new_attention_mask.append(torch.cat([before_mask, img_mask, after_mask]))

                # Update labels
                if labels is not None:
                    before_labels = labels[i, :img_pos]
                    after_labels = labels[i, img_pos + 1:]
                    # -100 for image tokens (don't compute loss)
                    img_labels = torch.full((self.config.num_vision_tokens,), -100, device=device)
                    new_labels.append(torch.cat([before_labels, img_labels, after_labels]))
            else:
                # No image token, use text as-is
                new_embeds.append(text_embeds[i])
                new_attention_mask.append(attention_mask[i])
                if labels is not None:
                    new_labels.append(labels[i])

        # Pad to same length
        max_len = max(e.shape[0] for e in new_embeds)

        padded_embeds = torch.zeros(batch_size, max_len, text_embeds.shape[-1], device=device, dtype=text_embeds.dtype)
        padded_attention = torch.zeros(batch_size, max_len, device=device, dtype=attention_mask.dtype)
        padded_labels = torch.full((batch_size, max_len), -100, device=device, dtype=torch.long) if labels is not None else None

        for i, (emb, mask) in enumerate(zip(new_embeds, new_attention_mask)):
            padded_embeds[i, :emb.shape[0]] = emb
            padded_attention[i, :mask.shape[0]] = mask
            if labels is not None:
                padded_labels[i, :new_labels[i].shape[0]] = new_labels[i]

        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=padded_embeds,
            attention_mask=padded_attention,
            labels=padded_labels,
            return_dict=True,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        questions: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[str]:
        """
        Generate answers for visual questions.

        Args:
            images: (batch, C, Z, H, W) cell images
            questions: List of question strings
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            List of generated answer strings
        """
        self.eval()
        batch_size = images.shape[0]
        device = images.device

        # Encode images
        vision_features = self.encode_images(images)
        image_embeds = self.projection(vision_features)

        # Prepare prompts
        prompts = [f"{self.image_token}\nQuestion: {q}\nAnswer:" for q in questions]

        # Tokenize
        encoded = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
        ).to(device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Get text embeddings and insert image embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        image_token_mask = (input_ids == self.image_token_id)

        new_embeds = []
        new_attention_mask = []

        for i in range(batch_size):
            img_positions = torch.where(image_token_mask[i])[0]

            if len(img_positions) > 0:
                img_pos = img_positions[0].item()
                before_img = text_embeds[i, :img_pos]
                after_img = text_embeds[i, img_pos + 1:]
                combined = torch.cat([before_img, image_embeds[i], after_img], dim=0)
                new_embeds.append(combined)

                before_mask = attention_mask[i, :img_pos]
                after_mask = attention_mask[i, img_pos + 1:]
                img_mask = torch.ones(self.config.num_vision_tokens, device=device)
                new_attention_mask.append(torch.cat([before_mask, img_mask, after_mask]))
            else:
                new_embeds.append(text_embeds[i])
                new_attention_mask.append(attention_mask[i])

        # Pad
        max_len = max(e.shape[0] for e in new_embeds)
        padded_embeds = torch.zeros(batch_size, max_len, text_embeds.shape[-1], device=device, dtype=text_embeds.dtype)
        padded_attention = torch.zeros(batch_size, max_len, device=device, dtype=attention_mask.dtype)

        for i, (emb, mask) in enumerate(zip(new_embeds, new_attention_mask)):
            padded_embeds[i, :emb.shape[0]] = emb
            padded_attention[i, :mask.shape[0]] = mask

        # Generate
        outputs = self.llm.generate(
            inputs_embeds=padded_embeds,
            attention_mask=padded_attention,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract answers (after "Answer:")
        answers = []
        for text in generated_texts:
            if "Answer:" in text:
                answer = text.split("Answer:")[-1].strip()
            else:
                answer = text.strip()
            answers.append(answer)

        return answers

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        params = []

        # Projection is always trainable
        params.extend(self.projection.parameters())

        # Vision encoder if not frozen
        if not self.config.freeze_vision_encoder:
            params.extend(self.vision_encoder.parameters())

        # LLM trainable parameters (LoRA or full)
        for name, param in self.llm.named_parameters():
            if param.requires_grad:
                params.append(param)

        return params

    def save_pretrained(self, save_path: str):
        """Save model checkpoint."""
        import os
        os.makedirs(save_path, exist_ok=True)

        # Save projection
        torch.save(
            self.projection.state_dict(),
            os.path.join(save_path, "projection.pt")
        )

        # Save LoRA weights if applicable
        if self.config.use_lora:
            self.llm.save_pretrained(os.path.join(save_path, "llm_lora"))

        # Save config
        import json
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

        print(f"Saved VLM to {save_path}")

    def load_pretrained(self, load_path: str):
        """Load model checkpoint."""
        import os

        # Load projection
        proj_path = os.path.join(load_path, "projection.pt")
        if os.path.exists(proj_path):
            self.projection.load_state_dict(torch.load(proj_path, map_location="cpu"))
            print(f"Loaded projection from {proj_path}")

        # Load LoRA weights if applicable
        lora_path = os.path.join(load_path, "llm_lora")
        if os.path.exists(lora_path) and self.config.use_lora:
            from peft import PeftModel
            self.llm = PeftModel.from_pretrained(self.llm, lora_path)
            print(f"Loaded LoRA from {lora_path}")


def build_vlm(
    mae_config,
    mae_checkpoint: str,
    llm_name: str = "mistralai/Mistral-7B-v0.1",
    freeze_vision: bool = True,
    freeze_llm: bool = True,
    use_lora: bool = True,
    num_vision_tokens: int = 64,
    device_map: str = "auto",
    model_type: str = "3d",
) -> CellVLM:
    """
    Build VLM model with all components.

    Args:
        mae_config: Config for MAE model
        mae_checkpoint: Path to pretrained MAE checkpoint
        llm_name: Name of LLM to load
        freeze_vision: Whether to freeze vision encoder
        freeze_llm: Whether to freeze LLM (ignored if use_lora=True)
        use_lora: Whether to use LoRA adapters
        num_vision_tokens: Number of visual tokens
        device_map: Device mapping for LLM
        model_type: "2d" or "3d" for MAE encoder type

    Returns:
        CellVLM model
    """
    # Determine vision hidden size from config
    # First check if encoder_embed_dim is specified (actual dimension)
    if hasattr(mae_config, 'encoder_embed_dim'):
        vision_hidden_size = mae_config.encoder_embed_dim
    else:
        # Fall back to arch-based heuristic
        arch = getattr(mae_config, 'arch', 'vit_base')
        if 'base' in arch:
            vision_hidden_size = 768
        elif 'large' in arch:
            vision_hidden_size = 1024
        elif 'huge' in arch:
            vision_hidden_size = 1280
        else:
            vision_hidden_size = 768

    print(f"Vision hidden size: {vision_hidden_size}")

    # Determine LLM hidden size
    if 'mistral' in llm_name.lower() or 'llama' in llm_name.lower():
        llm_hidden_size = 4096
    else:
        llm_hidden_size = 4096  # Default

    config = VLMConfig(
        vision_hidden_size=vision_hidden_size,
        vision_checkpoint=mae_checkpoint,
        freeze_vision_encoder=freeze_vision,
        llm_name=llm_name,
        llm_hidden_size=llm_hidden_size,
        freeze_llm=freeze_llm,
        use_lora=use_lora,
        num_vision_tokens=num_vision_tokens,
    )

    model = CellVLM(config)
    model.load_vision_encoder(mae_config, mae_checkpoint, model_type=model_type)
    model.load_llm(device_map=device_map)

    return model
