"""Fine-tunes the local model on your own handwriting (LoRA) using pages you verified and your samples.

Stop the server first (the GPU can't hold two copies of the model), then run:
    python -m app.train            # or train.bat on Windows
Options: --epochs 3 --lr 1e-4 --rank 16 --synth 1500 --no-eval --fresh

Synthetic lines from synth.py (synth.bat) are mixed in: a fresh random share of them each epoch.
"""

import argparse
import random
import shutil
import time

from PIL import Image, ImageEnhance

from . import local_model, storage, synth
from .ocr import BASE_PROMPT


def collect_examples() -> list[tuple]:
    examples = [(storage.page_image_path(p["id"]), p["text"]) for p in storage.list_pages()
                if p["status"] == "verified" and p["text"].strip()]
    examples += [(storage.sample_image_path(s["id"]), s["label"]) for s in storage.list_samples() if s["label"].strip()]
    return examples


def synthetic_examples() -> list[tuple]:
    return [(synth.line_image_path(l["id"]), l["label"]) for l in synth.list_lines()
            if synth.line_image_path(l["id"]).exists()]


def augment(img: Image.Image) -> Image.Image:
    """Small changes in angle and lighting, like different photos of the same page."""
    img = img.rotate(random.uniform(-2, 2), expand=True, fillcolor="white", resample=Image.BICUBIC)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
    return ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.2))


def encode(processor, image, answer: str):
    """Tokenises one example; only the transcription tokens count towards the loss."""
    full = processor.apply_chat_template(
        local_model.build_messages(image, BASE_PROMPT, answer),
        tokenize=True, return_dict=True, return_tensors="pt",
    )
    prompt = processor.apply_chat_template(
        local_model.build_messages(image, BASE_PROMPT),
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    )
    prompt_len = prompt["input_ids"].shape[1]
    labels = full["input_ids"].clone()
    labels[:, :prompt_len] = -100
    full["labels"] = labels
    return full


def cer(predicted: str, truth: str) -> float:
    """Character error rate: share of characters that would have to be fixed."""
    import difflib

    sm = difflib.SequenceMatcher(a=predicted, b=truth, autojunk=False)
    errors = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
    return errors / max(len(truth), 1)


def evaluate(model, processor, examples) -> float:
    import torch

    model.eval()
    scores = []
    for path, truth in examples:
        image = Image.open(path).convert("RGB")
        inputs = processor.apply_chat_template(
            local_model.build_messages(image, BASE_PROMPT),
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=local_model.MAX_NEW_TOKENS, do_sample=False)
        text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        text = text.replace("{{", "").replace("}}", "")
        scores.append(cer(text.strip(), truth.strip()))
    return sum(scores) / len(scores)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    ap.add_argument("--no-eval", action="store_true", help="skip measuring accuracy before/after")
    ap.add_argument("--fresh", action="store_true", help="start from the base model instead of the last adapter")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (for testing)")
    ap.add_argument("--synth", type=int, default=-1,
                    help="synthetic lines per epoch (default: twice your own examples, at most 1500; 0 = none)")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    examples = collect_examples()
    if len(examples) < 5:
        raise SystemExit(f"Мало данных: {len(examples)} примеров. Проверьте и сохраните хотя бы 5 страниц или образцов.")
    random.seed(0)
    random.shuffle(examples)
    pages_only = [e for e in examples if e[0].name == "image.jpg"]
    holdout = [] if args.no_eval or len(pages_only) < 5 else pages_only[: max(1, min(5, len(pages_only) // 10))]
    train_set = [e for e in examples if e not in holdout]
    synthetic = synthetic_examples()
    per_epoch = args.synth if args.synth >= 0 else min(1500, 2 * len(train_set))
    per_epoch = min(per_epoch, len(synthetic))
    print(f"Примеров для обучения: {len(train_set)} своих + {per_epoch} синтетических за эпоху "
          f"(всего синтетических {len(synthetic)}), для проверки точности: {len(holdout)}")
    print(f"Модель: {local_model.model_name()} на {local_model.device()}")

    processor = local_model.load_processor()
    model = local_model.load_base_model(for_training=True)
    if local_model.device() == "cuda":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    resume = local_model.has_adapter() and not args.fresh
    if resume:
        print("Продолжаю обучение с прошлого адаптера")
        model = PeftModel.from_pretrained(model, str(local_model.ADAPTER_DIR), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
            # Only the language part: it learns which Lithuanian words your strokes spell.
            target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
        ))
    model.print_trainable_parameters()

    if holdout:
        print("Измеряю точность до обучения…")
        before = evaluate(model, processor, holdout)
        print(f"Ошибок в символах до обучения: {before:.1%}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    epoch_size = len(train_set) + per_epoch
    total_steps = args.epochs * ((epoch_size + args.accum - 1) // args.accum)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: max(0.0, 1 - s / max(total_steps, 1)))
    use_bf16 = local_model.device() == "cuda"

    model.train()
    step, started = 0, time.time()
    for epoch in range(args.epochs):
        epoch_set = train_set + random.sample(synthetic, per_epoch)
        random.shuffle(epoch_set)
        running = 0.0
        for i, (path, text) in enumerate(epoch_set, 1):
            batch = encode(processor, augment(Image.open(path).convert("RGB")), text).to(model.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**batch).loss / args.accum
            loss.backward()
            running += loss.item() * args.accum
            if i % args.accum == 0 or i == len(epoch_set):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                print(f"эпоха {epoch + 1}/{args.epochs}  шаг {step}/{total_steps}  loss {running / i:.3f}  "
                      f"{(time.time() - started) / 60:.1f} мин", flush=True)
                if args.max_steps and step >= args.max_steps:
                    break
        if args.max_steps and step >= args.max_steps:
            break

    if holdout:
        after = evaluate(model, processor, holdout)
        print(f"Ошибок в символах: было {before:.1%}, стало {after:.1%}")
        if after > before:
            print("Обучение не улучшило точность, адаптер не сохраняю. Попробуйте --lr 5e-5 или больше данных.")
            return

    tmp = local_model.ADAPTER_DIR.with_name("adapter_new")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(str(tmp))
    if local_model.ADAPTER_DIR.exists():
        prev = local_model.ADAPTER_DIR.with_name("adapter_prev")
        shutil.rmtree(prev, ignore_errors=True)
        local_model.ADAPTER_DIR.rename(prev)
    tmp.rename(local_model.ADAPTER_DIR)
    print(f"Готово. Адаптер сохранён в {local_model.ADAPTER_DIR}. Запустите сервер заново.")


if __name__ == "__main__":
    main()
