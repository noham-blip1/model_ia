
# -*- coding: utf-8 -*-
"""Imagine_v1 - IANC-003 : inference robuste."""

from pathlib import Path
import hashlib
import torch
from tokenizers import Tokenizer

from ianc003_common import build_model_from_checkpoint, sha256_file


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_FILE = ROOT / "models" / "ianc003_refined" / "ianc003_refined_best.pt"
TOKENIZER_FILE = ROOT / "models" / "ianc002" / "tokenizer.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_NEW_TOKENS = 120
TEMPERATURE = 0.70
TOP_K = 40
REPETITION_PENALTY = 1.12
REPETITION_WINDOW = 64

# False = génération déterministe, idéale pour les tests.
# True = génération avec échantillonnage.
DO_SAMPLE = False

SUPPORTED_FORMATS = {
    "segments_v1_user_eos_imagine_v1",
    "segments_v2_masked_role_prefix_user_eos",
}


def load_model():
    if not CHECKPOINT_FILE.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable : {CHECKPOINT_FILE}")

    if not TOKENIZER_FILE.is_file():
        raise FileNotFoundError(f"Tokenizer introuvable : {TOKENIZER_FILE}")

    print("[LOAD] Chargement du tokenizer...")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))

    vocab_size = tokenizer.get_vocab_size()
    eos_id = tokenizer.token_to_id("[EOS]")

    if eos_id is None:
        raise RuntimeError("Token [EOS] absent du tokenizer.")

    print("[LOAD] Chargement du checkpoint...")
    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=True
    )

    prompt_format = checkpoint.get("prompt_format")

    if prompt_format not in SUPPORTED_FORMATS:
        raise RuntimeError(
            f"Format de prompt non pris en charge : {prompt_format}"
        )

    expected_hash = (
        checkpoint.get("tokenizer_sha256")
        or checkpoint.get("tokenizer_hash")
    )

    actual_hash = sha256_file(TOKENIZER_FILE)

    if expected_hash and expected_hash != actual_hash:
        raise RuntimeError(
            "Le tokenizer ne correspond pas au checkpoint."
        )

    model, config = build_model_from_checkpoint(
        checkpoint,
        vocab_size
    )

    model.to(DEVICE)
    model.eval()

    block_size = int(config["block_size"])

    if block_size < 2:
        raise RuntimeError("Block size invalide.")

    print("[OK] Modèle chargé.")
    print("Device :", DEVICE)
    print("Vocabulaire :", vocab_size)
    print("Block size :", block_size)
    print("Étape du checkpoint :", checkpoint.get("step", "?"))
    print("Format :", prompt_format)

    if DEVICE.type == "cuda":
        print("GPU :", torch.cuda.get_device_name(0))

    return model, tokenizer, eos_id, block_size, vocab_size


def build_prompt_ids(tokenizer, message, eos_id):
    message = message.strip()

    if not message:
        raise ValueError("Le message utilisateur est vide.")

    prompt = "\nUtilisateur : " + message

    ids = tokenizer.encode(prompt).ids
    ids.append(eos_id)

    assistant_prefix = tokenizer.encode(
        "\nImagine_v1 : "
    ).ids

    ids.extend(assistant_prefix)

    if not ids:
        raise RuntimeError("Le prompt ne contient aucun token.")

    return ids


def apply_repetition_penalty(logits, recent_tokens):
    if REPETITION_PENALTY <= 1.0:
        return logits

    for token_id in set(recent_tokens):
        if token_id < 0 or token_id >= logits.shape[-1]:
            continue

        value = logits[token_id]

        if value < 0:
            logits[token_id] *= REPETITION_PENALTY
        else:
            logits[token_id] /= REPETITION_PENALTY

    return logits


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt_ids,
    eos_id,
    block_size,
    vocab_size,
    do_sample=False
):
    generated = []
    current_ids = list(prompt_ids)

    for _ in range(MAX_NEW_TOKENS):
        context = current_ids[-block_size:]

        x = torch.tensor(
            [context],
            dtype=torch.long,
            device=DEVICE
        )

        output = model(x)

        if isinstance(output, (tuple, list)):
            logits = output[0]
        else:
            logits = output

        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("Le modèle n'a pas retourné de Tensor.")

        if logits.ndim != 3:
            raise RuntimeError(
                f"Dimensions des logits invalides : {tuple(logits.shape)}"
            )

        if logits.shape[0] != 1:
            raise RuntimeError("Batch inattendu.")

        if logits.shape[-1] != vocab_size:
            raise RuntimeError("Taille du vocabulaire incorrecte.")

        next_logits = logits[0, -1, :].float().clone()

        if not torch.isfinite(next_logits).all():
            raise RuntimeError("Logits non finis détectés.")

        recent = current_ids[-REPETITION_WINDOW:]
        next_logits = apply_repetition_penalty(
            next_logits,
            recent
        )

        if do_sample:
            if TEMPERATURE <= 0:
                raise RuntimeError("Température invalide.")

            next_logits /= TEMPERATURE

            if TOP_K > 0:
                k = min(TOP_K, next_logits.numel())
                threshold = torch.topk(next_logits, k).values[-1]
                next_logits[next_logits < threshold] = float("-inf")

            probabilities = torch.softmax(next_logits, dim=-1)

            if not torch.isfinite(probabilities).all():
                raise RuntimeError("Probabilités invalides.")

            if probabilities.sum() <= 0:
                raise RuntimeError("Somme des probabilités nulle.")

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            ).item()

        else:
            next_token = torch.argmax(next_logits).item()

        if next_token == eos_id:
            break

        generated.append(next_token)
        current_ids.append(next_token)

    if not generated:
        return ""

    result = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    return result.strip()


def main():
    print("=" * 55)
    print("Imagine_v1 — Systeme initialise")
    print("IANC-003")
    print("=" * 55)

    model, tokenizer, eos_id, block_size, vocab_size = load_model()

    print("\nMode :", "échantillonné" if DO_SAMPLE else "déterministe")
    print("Commandes : /quit pour quitter.")
    print()

    while True:
        try:
            message = input("Vous : ").strip()

            if not message:
                continue

            if message.lower() in ("/quit", "/exit", "/stop"):
                print("Imagine_v1 : Arrêt du système.")
                break

            prompt_ids = build_prompt_ids(
                tokenizer,
                message,
                eos_id
            )

            if len(prompt_ids) > block_size:
                prefix = tokenizer.encode(
                    "\nImagine_v1 : "
                ).ids

                body = prompt_ids[:-len(prefix)]
                prompt_ids = body[-(block_size - len(prefix)):] + prefix

            response = generate(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                eos_id=eos_id,
                block_size=block_size,
                vocab_size=vocab_size,
                do_sample=DO_SAMPLE
            )

            if response:
                print("\nImagine_v1 :", response, "\n")
            else:
                print("\nImagine_v1 : [Aucune réponse générée]\n")

        except KeyboardInterrupt:
            print("\nImagine_v1 : Arrêt demandé.")
            break

        except Exception as error:
            print("\n[ERREUR]", error)


if __name__ == "__main__":
    main()
