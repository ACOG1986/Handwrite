"""Local vision-language model (Qwen-VL via transformers). Runs on an NVIDIA GPU, Apple Silicon or CPU."""

import os
from pathlib import Path

from . import gpu, storage

# On a 12 GB card the 8B model fits in 4-bit (~6 GB); on smaller machines use the 2B model.
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
SMALL_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

ADAPTER_DIR = storage.DATA_DIR / "adapter"  # LoRA weights trained on the user's handwriting (train.py)

# Caps how many image tokens a page becomes: more = sharper reading, but slower and more memory.
MAX_PIXELS = int(os.environ.get("HANDAPP_MAX_PIXELS", str(1600 * 28 * 28)))
MAX_NEW_TOKENS = int(os.environ.get("HANDAPP_MAX_NEW_TOKENS", "2048"))

_lock = gpu.lock  # shared with the proofreader (context_fix.py): one model on the GPU at a time
_model = None
_processor = None


def device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_name() -> str:
    if os.environ.get("HANDAPP_MODEL"):
        return os.environ["HANDAPP_MODEL"]
    return DEFAULT_MODEL if device() == "cuda" else SMALL_MODEL


def load_processor(name: str | None = None):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(name or model_name())
    processor.image_processor.size["longest_edge"] = MAX_PIXELS  # size keys count pixels here, not sides
    return processor


def load_base_model(name: str | None = None, for_training: bool = False):
    import torch
    from transformers import AutoModelForImageTextToText

    name = name or model_name()
    kwargs = {}
    if device() == "cuda":
        quant = os.environ.get("HANDAPP_QUANT", "4bit")
        if quant == "4bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = "auto"
    elif device() == "mps":
        kwargs["dtype"] = torch.float16 if not for_training else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(name, **kwargs)
    if device() == "mps":
        model.to("mps")
    return model


def _load():
    global _model, _processor
    if _model is not None:
        return
    model = load_base_model()
    if (ADAPTER_DIR / "adapter_config.json").exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    model.eval()
    _processor = load_processor()
    _model = model


def unload():
    """Free the GPU for the proofreader. Call with the GPU lock held."""
    global _model, _processor
    if _model is None:
        return
    _model = _processor = None
    gpu.free_cache()


gpu.register("reader", unload)


def reload():
    """Drop the loaded model so the next request picks up a freshly trained adapter."""
    with _lock:
        unload()


def is_loaded() -> bool:
    return _model is not None


def has_adapter() -> bool:
    return (ADAPTER_DIR / "adapter_config.json").exists()


def build_messages(image, prompt: str, answer: str | None = None) -> list[dict]:
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def generate(image_path: Path, prompt: str) -> str:
    import torch
    from PIL import Image

    with gpu.use("reader"):  # one page at a time: the GPU can't hold two
        _load()
        image = Image.open(image_path).convert("RGB")
        inputs = _processor.apply_chat_template(
            build_messages(image, prompt),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(_model.device)
        with torch.inference_mode():
            out = _model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.05)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return _processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
