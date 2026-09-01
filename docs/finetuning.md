# Fine-tuning

Diffusion fine-tuning uses LoRA adapters on the UNet attention layers.
Ranks 8–16 give the best quality/size tradeoff for this task. QLoRA is used
for the LLM head with 4-bit base weights.

Training runs are tracked in Weights & Biases (project `genmm-a11y`).
