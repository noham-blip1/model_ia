# -*- coding: utf-8 -*-

from pathlib import Path
import hashlib
import json
import random
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from ianc003_common import sha256_file, build_model_from_checkpoint


# ---------------------------------------------------------------------------
# CHEMINS
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

CLEAN_DIR = ROOT / "data" / "ianc003_clean"
TRAIN_FILE = CLEAN_DIR / "train_clean.jsonl"
VAL_FILE = CLEAN_DIR / "val_clean.jsonl"

TOKENIZER_FILE = ROOT / "models" / "ianc002" / "tokenizer.json"
SOURCE_CHECKPOINT = ROOT / "models" / "ianc003" / "ianc003_best.pt"

MODEL_DIR = ROOT / "models" / "ianc003_refined"
BEST_CHECKPOINT = MODEL_DIR / "ianc003_refined_best.pt"
LATEST_STATE = MODEL_DIR / "refined_training_state.pt"
METRICS_FILE = MODEL_DIR / "refined_metrics.json"

MAX_STEPS = 1200
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

EVAL_INTERVAL = 100
EVAL_BATCHES = 32
PATIENCE = 6
SEED = 2026

TRAINING_FORMAT = "ianc003_clean_train_val_v3"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"


# ---------------------------------------------------------------------------
# OUTILS
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True


def normalize_conversation(item):
    """Accepte les conversations JSONL utilisees par l'audit."""
    if not isinstance(item, dict):
        return None

    messages = item.get("messages", item.get("conversation"))
    if not isinstance(messages, list):
        return None

    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            continue

        role = str(message.get("role", "")).strip().lower()
        content = message.get("content", message.get("text", ""))
        content = str(content or "").strip()

        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant", "bot"):
            role = "assistant"
        else:
            continue

        if content:
            normalized.append({"role": role, "content": content})

    if len(normalized) < 2:
        return None
    if normalized[0]["role"] != "user" or normalized[-1]["role"] != "assistant":
        return None

    for index, message in enumerate(normalized):
        expected = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected:
            return None

    return {
        "context": str(item.get("context", "") or "").strip(),
        "messages": normalized,
    }


def load_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(f"Jeu de donnees introuvable : {path}")

    rows = []
    invalid = 0
    duplicates = 0
    seen = set()

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue

            item = normalize_conversation(raw)
            if item is None:
                invalid += 1
                continue

            signature = json.dumps(
                item, ensure_ascii=False, sort_keys=True,
                separators=(",", ":")
            )
            digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()

            if digest in seen:
                duplicates += 1
                continue

            seen.add(digest)
            rows.append(item)

    if len(rows) < 20:
        raise RuntimeError(
            f"Pas assez de conversations valides dans {path.name} : {len(rows)}"
        )

    print(f"\n[DATASET] {path.name}")
    print("Conversations valides :", len(rows))
    print("Lignes invalides      :", invalid)
    print("Doublons              :", duplicates)

    return rows


def encode_conversation(example, tokenizer, eos_id):
    ids = []
    labels = []

    context = example["context"]
    if context:
        context_ids = (
            tokenizer.encode("Contexte : ").ids
            + tokenizer.encode(context).ids
        )
        ids.extend(context_ids)
        labels.extend([-100] * len(context_ids))

    for message in example["messages"]:
        if message["role"] == "user":
            prefix = tokenizer.encode("\nUtilisateur : ").ids
            content = tokenizer.encode(message["content"]).ids
            segment = prefix + content

            ids.extend(segment)
            labels.extend([-100] * len(segment))

            ids.append(eos_id)
            labels.append(-100)

        else:
            prefix = tokenizer.encode("\nImagine_v1 : ").ids
            content = tokenizer.encode(message["content"]).ids

            ids.extend(prefix)
            labels.extend([-100] * len(prefix))

            ids.extend(content)
            labels.extend(content)

            ids.append(eos_id)
            labels.append(eos_id)

    return {"input_ids": ids, "labels": labels}


def prepare_data(examples, tokenizer, eos_id, split_name):
    encoded = []

    for example in examples:
        row = encode_conversation(example, tokenizer, eos_id)
        if len(row["input_ids"]) >= 2 and any(
            value != -100 for value in row["labels"][1:]
        ):
            encoded.append(row)

    if len(encoded) < 20:
        raise RuntimeError(
            f"Pas assez de conversations encodables dans {split_name}."
        )

    print(f"[ENCODAGE] {split_name} : {len(encoded)} conversations")
    return encoded


def sample_one(dataset, block_size, rng, pad_id=0):
    """Construit X et Y decales d'un token, avec masquage des prompts."""
    for _ in range(200):
        example = rng.choice(dataset)
        ids = example["input_ids"]
        labels = example["labels"]

        valid_targets = [
            index for index in range(1, len(labels))
            if labels[index] != -100
        ]
        if not valid_targets:
            continue

        target = rng.choice(valid_targets)
        start_min = max(0, target - block_size + 1)
        start_max = target - 1

        if start_min > start_max:
            continue

        start = rng.randint(start_min, start_max)
        end = min(start + block_size + 1, len(ids))

        x = ids[start:end - 1]
        y = labels[start + 1:end]

        if not any(value != -100 for value in y):
            continue

        if len(x) < block_size:
            padding = block_size - len(x)
            x += [pad_id] * padding
            y += [-100] * padding

        return x, y

    raise RuntimeError("Impossible de construire un echantillon d'entrainement.")


def make_fixed_eval(dataset, block_size, count, seed):
    rng = random.Random(seed)
    samples = [
        sample_one(dataset, block_size, rng)
        for _ in range(count)
    ]

    x_tensor = torch.tensor(
        [x for x, _ in samples], dtype=torch.long, device=DEVICE
    )
    y_tensor = torch.tensor(
        [y for _, y in samples], dtype=torch.long, device=DEVICE
    )
    return x_tensor, y_tensor


@torch.no_grad()
def evaluate(model, fixed_batch):
    model.eval()
    x, y = fixed_batch

    with torch.autocast(
        device_type=DEVICE.type,
        dtype=torch.float16,
        enabled=AMP_ENABLED
    ):
        _, loss = model(x, y)

    model.train()
    return float(loss.item())


def save_checkpoint(
    model, config, vocab_size, global_step, local_step, val_loss,
    tokenizer_hash, dataset_hash, base_hash
):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "vocab_size": int(vocab_size),
        "config": dict(config),
        "step": int(global_step),
        "finetune_step": int(local_step),
        "val_loss": float(val_loss),
        "best_validation_loss": float(val_loss),
        "tokenizer_hash": tokenizer_hash,
        "tokenizer_sha256": tokenizer_hash,
        "dataset_hash": dataset_hash,
        "base_checkpoint_hash": base_hash,
        "prompt_format": "segments_v2_masked_role_prefix_user_eos",
        "training_format": TRAINING_FORMAT,
    }

    temporary = BEST_CHECKPOINT.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(BEST_CHECKPOINT)

    print("[SAVE] Meilleur checkpoint :", BEST_CHECKPOINT)


def save_state(
    model, optimizer, scaler, train_rng, local_step, best_val, best_step,
    patience_counter, tokenizer_hash, dataset_hash, base_hash, run_hash
):
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "step": int(local_step),
        "best_val": float(best_val),
        "best_step": int(best_step),
        "patience_counter": int(patience_counter),
        "tokenizer_hash": tokenizer_hash,
        "dataset_hash": dataset_hash,
        "base_checkpoint_hash": base_hash,
        "run_hash": run_hash,
        "training_format": TRAINING_FORMAT,
        "python_rng_state": random.getstate(),
        "train_rng_state": train_rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else []
        ),
    }

    temporary = LATEST_STATE.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(LATEST_STATE)


# ---------------------------------------------------------------------------
# ENTRAINEMENT
# ---------------------------------------------------------------------------

def train():
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    required_files = (
        TRAIN_FILE, VAL_FILE, TOKENIZER_FILE, SOURCE_CHECKPOINT
    )
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Fichier requis introuvable : {path}")

    print("=" * 68)
    print("IMAGINE_V1 — IANC-003 CLEAN FINE-TUNING")
    print("MonarqueOS | Build the rules. Build the world.")
    print("=" * 68)

    print("Device :", DEVICE)
    if AMP_ENABLED:
        properties = torch.cuda.get_device_properties(0)
        print("GPU    :", torch.cuda.get_device_name(0))
        print("VRAM   :", round(properties.total_memory / 1024**3, 2), "Go")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    vocab_size = tokenizer.get_vocab_size()
    eos_id = tokenizer.token_to_id("[EOS]")

    if eos_id is None:
        raise RuntimeError("Le token special [EOS] est absent du tokenizer.")

    tokenizer_hash = sha256_file(TOKENIZER_FILE)
    train_hash = sha256_file(TRAIN_FILE)
    val_hash = sha256_file(VAL_FILE)
    dataset_hash = hashlib.sha256(
        (train_hash + val_hash).encode("ascii")
    ).hexdigest()
    base_hash = sha256_file(SOURCE_CHECKPOINT)

    print("\n[BASE] Chargement du checkpoint IANC-003...")
    base_checkpoint = torch.load(
        SOURCE_CHECKPOINT, map_location="cpu", weights_only=True
    )
    model, config = build_model_from_checkpoint(
        base_checkpoint, vocab_size
    )

    block_size = int(config["block_size"])
    base_step = int(base_checkpoint.get("step", 0))

    model.to(DEVICE)

    print("Checkpoint initial :", SOURCE_CHECKPOINT)
    print("Etape initiale     :", base_step)
    print("Parametres         :", sum(p.numel() for p in model.parameters()))
    print("Block size         :", block_size)

    print("\n[DATA] Chargement des donnees nettoyees...")
    train_examples = load_jsonl(TRAIN_FILE)
    val_examples = load_jsonl(VAL_FILE)

    train_data = prepare_data(
        train_examples, tokenizer, eos_id, "TRAIN"
    )
    val_data = prepare_data(
        val_examples, tokenizer, eos_id, "VALIDATION"
    )

    # Les evaluations utilisent des echantillons fixes et reproductibles.
    train_eval = make_fixed_eval(
        train_data, block_size, EVAL_BATCHES, SEED + 11
    )
    val_eval = make_fixed_eval(
        val_data, block_size, EVAL_BATCHES, SEED + 22
    )

    run_config = {
        "block_size": block_size,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "eval_interval": EVAL_INTERVAL,
        "eval_batches": EVAL_BATCHES,
        "patience": PATIENCE,
        "training_format": TRAINING_FORMAT,
    }
    run_hash = hashlib.sha256(
        json.dumps(run_config, sort_keys=True).encode("utf-8")
    ).hexdigest()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    start_step = 0
    best_val = float("inf")
    best_step = 0
    patience_counter = 0
    train_rng = random.Random(SEED + 100)

    # Reprise uniquement si les donnees, le tokenizer et la base sont identiques.
    if LATEST_STATE.is_file():
        print("\n[RESUME] Etat d'entrainement precedent detecte.")
        state = torch.load(
            LATEST_STATE, map_location="cpu", weights_only=True
        )

        compatible = (
            state.get("dataset_hash") == dataset_hash
            and state.get("tokenizer_hash") == tokenizer_hash
            and state.get("base_checkpoint_hash") == base_hash
            and state.get("run_hash") == run_hash
            and state.get("training_format") == TRAINING_FORMAT
        )

        if compatible:
            print("[RESUME] Reprise de l'entrainement.")
            model.load_state_dict(state["model_state"], strict=True)
            optimizer.load_state_dict(state["optimizer_state"])

            if state.get("scaler_state"):
                scaler.load_state_dict(state["scaler_state"])

            start_step = int(state.get("step", 0))
            best_val = float(state.get("best_val", float("inf")))
            best_step = int(state.get("best_step", 0))
            patience_counter = int(state.get("patience_counter", 0))

            try:
                random.setstate(state["python_rng_state"])
                train_rng.setstate(state["train_rng_state"])
                torch.set_rng_state(state["torch_rng_state"])

                if torch.cuda.is_available() and state.get("cuda_rng_state"):
                    torch.cuda.set_rng_state_all(state["cuda_rng_state"])
            except Exception as error:
                print("[WARN] Etat aleatoire non restaure :", error)
        else:
            print(
                "[RESUME] Etat incompatible : depart depuis le checkpoint "
                "de base. L'ancien etat est conserve jusqu'a la premiere sauvegarde."
            )

    if start_step >= MAX_STEPS:
        print(
            f"\n[INFO] {start_step} etapes deja effectuees "
            f"(limite {MAX_STEPS})."
        )
        return

    print("\n[TRAIN] Etapes supplementaires max :", MAX_STEPS)
    print("[TRAIN] Learning rate             :", LEARNING_RATE)
    print("[TRAIN] Batch effectif             :", BATCH_SIZE * GRAD_ACCUM)
    print("[TRAIN] Taille des sequences      :", block_size)
    print("[TRAIN] Validation fixe            :", EVAL_BATCHES, "echantillons")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    started = time.time()
    history = []

    if METRICS_FILE.is_file():
        try:
            previous = json.loads(
                METRICS_FILE.read_text(encoding="utf-8")
            )
            if isinstance(previous, list):
                history = previous
        except Exception:
            history = []

    for local_step in range(start_step + 1, MAX_STEPS + 1):
        accumulated_loss = 0.0

        for _ in range(GRAD_ACCUM):
            x_rows = []
            y_rows = []

            for _ in range(BATCH_SIZE):
                x, y = sample_one(train_data, block_size, train_rng)
                x_rows.append(x)
                y_rows.append(y)

            x_tensor = torch.tensor(
                x_rows, dtype=torch.long, device=DEVICE
            )
            y_tensor = torch.tensor(
                y_rows, dtype=torch.long, device=DEVICE
            )

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=AMP_ENABLED
            ):
                _, loss = model(x_tensor, y_tensor)
                scaled_loss = loss / GRAD_ACCUM

            scaler.scale(scaled_loss).backward()
            accumulated_loss += float(loss.detach().item()) / GRAD_ACCUM

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if local_step % 10 == 0 or local_step == start_step + 1:
            elapsed = time.time() - started
            print(
                f"Step {local_step:4d}/{MAX_STEPS} | "
                f"train loss {accumulated_loss:.4f} | {elapsed:.1f}s"
            )

        if local_step % EVAL_INTERVAL == 0 or local_step == MAX_STEPS:
            train_loss = evaluate(model, train_eval)
            val_loss = evaluate(model, val_eval)

            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                best_step = local_step
                patience_counter = 0

                save_checkpoint(
                    model, config, vocab_size,
                    base_step + local_step, local_step, val_loss,
                    tokenizer_hash, dataset_hash, base_hash
                )
                print("[BEST] Nouvelle meilleure validation :", round(best_val, 6))
            else:
                patience_counter += 1

            record = {
                "local_step": local_step,
                "global_step": base_step + local_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val": best_val,
                "best_step": best_step,
                "patience_counter": patience_counter,
                "learning_rate": LEARNING_RATE,
                "training_format": TRAINING_FORMAT,
            }
            history.append(record)

            temporary_metrics = METRICS_FILE.with_suffix(".tmp")
            temporary_metrics.write_text(
                json.dumps(history, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            temporary_metrics.replace(METRICS_FILE)

            save_state(
                model, optimizer, scaler, train_rng,
                local_step, best_val, best_step, patience_counter,
                tokenizer_hash, dataset_hash, base_hash, run_hash
            )

            print(
                f"[EVAL] step={local_step} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"best={best_val:.4f} | "
                f"patience={patience_counter}/{PATIENCE}"
            )

            if patience_counter >= PATIENCE:
                print("[STOP] Early stopping : la validation ne s'ameliore plus.")
                break

    print("\n" + "=" * 68)
    print("FINE-TUNING TERMINE")
    print("Meilleure etape locale :", best_step)
    print("Etape globale approx.  :", base_step + best_step)
    print("Meilleure val loss     :", best_val)
    print("Checkpoint affine      :", BEST_CHECKPOINT)
    print("Duree                  :", round(time.time() - started, 1), "s")
    print("=" * 68)


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print(
            "\n[STOP] Entrainement interrompu. "
            "Le dernier etat sauvegarde a chaque evaluation est conserve."
        )
    except Exception as error:
        print("\n[ERREUR]", error)
        raise
