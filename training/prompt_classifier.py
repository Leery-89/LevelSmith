"""
Prompt → graph_family / intent classifier for LevelSmith.

Loads the trained classifier (training/classifier/classifier_best.pt) and
provides a single `classify_prompt(prompt, style=None)` function.

The classifier is loaded lazily on first call so importing this module is
cheap. If the model files are missing or torch/sentence-transformers are not
installed, `classify_prompt()` returns None and the caller falls back to its
existing path.

When prediction confidence is below CONFIDENCE_THRESHOLD and a DeepSeek API
key is configured, the prompt is sent to the LLM for confirmation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

# ─── Paths and config ────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
CLASSIFIER_DIR = SCRIPT_DIR / "classifier"
MODEL_PATH = CLASSIFIER_DIR / "classifier_best.pt"
LABEL_MAPS_PATH = CLASSIFIER_DIR / "label_maps.json"

CONFIDENCE_THRESHOLD = 0.6

# Load training/.env if present so DEEPSEEK_API_KEY works in standalone scripts
try:
    from dotenv import load_dotenv
    load_dotenv(SCRIPT_DIR / ".env")
except ImportError:
    pass


# ─── Lazy state ──────────────────────────────────────────────────────

_state: dict = {
    "loaded": False,        # True after first load attempt (success or fail)
    "available": False,     # True only if model + deps loaded successfully
    "model": None,
    "encoder": None,
    "families": None,
    "intents": None,
    "styles": None,
    "style_to_idx": None,
    "embed_dim": None,
    "style_dim": None,
    "head_type": None,
    "load_error": None,
}


def _build_model(embed_dim: int, style_dim: int, num_families: int,
                 num_intents: int, head_type: str):
    """Reconstruct the classifier architecture used during training."""
    import torch.nn as nn

    class GraphFamilyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.use_style = style_dim > 0
            input_dim = embed_dim + style_dim
            if head_type == "linear":
                self.shared = nn.Identity()
                self.family_head = nn.Linear(input_dim, num_families)
                self.intent_head = nn.Linear(input_dim, num_intents)
            else:  # mlp
                self.shared = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.GELU(),
                    nn.Dropout(0.15),
                    nn.Linear(128, 64),
                    nn.GELU(),
                    nn.Dropout(0.10),
                )
                self.family_head = nn.Linear(64, num_families)
                self.intent_head = nn.Linear(64, num_intents)

        def forward(self, text_emb, style_onehot=None):
            import torch
            if self.use_style and style_onehot is not None:
                x = torch.cat([text_emb, style_onehot], dim=-1)
            else:
                x = text_emb
            x = self.shared(x)
            return {
                "family": self.family_head(x),
                "intent": self.intent_head(x),
            }

    return GraphFamilyClassifier()


def _load() -> bool:
    """Load classifier on first call. Returns True if available."""
    if _state["loaded"]:
        return _state["available"]
    _state["loaded"] = True

    if not MODEL_PATH.exists() or not LABEL_MAPS_PATH.exists():
        _state["load_error"] = (
            f"classifier files not found at {CLASSIFIER_DIR}")
        return False

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        _state["load_error"] = f"missing dependency: {e}"
        return False

    try:
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    except Exception as e:
        _state["load_error"] = f"failed to load checkpoint: {e}"
        return False

    label_maps = json.loads(LABEL_MAPS_PATH.read_text(encoding="utf-8"))
    families = ckpt.get("families") or label_maps["families"]
    intents = ckpt.get("intents") or label_maps["intents"]
    styles = ckpt.get("styles") or label_maps.get("styles", [])
    style_to_idx = {s: i for i, s in enumerate(styles)}
    embed_dim = ckpt["embed_dim"]
    style_dim = ckpt["style_dim"]
    head_type = ckpt["head_type"]
    encoder_name = ckpt["encoder"]

    try:
        encoder = SentenceTransformer(encoder_name, device="cpu")
    except Exception as e:
        _state["load_error"] = f"failed to load encoder {encoder_name}: {e}"
        return False

    model = _build_model(embed_dim, style_dim, len(families), len(intents),
                         head_type)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _state.update({
        "available": True,
        "model": model,
        "encoder": encoder,
        "families": families,
        "intents": intents,
        "styles": styles,
        "style_to_idx": style_to_idx,
        "embed_dim": embed_dim,
        "style_dim": style_dim,
        "head_type": head_type,
    })
    return True


# ─── DeepSeek confirmation (low-confidence fallback) ─────────────────

def _confirm_with_deepseek(prompt: str, top_candidates: list[tuple[str, float]],
                           top_intent: str) -> Optional[dict]:
    """Ask DeepSeek to confirm or correct the classifier output.

    top_candidates: list of (family_name, prob) sorted by prob desc.
    Returns updated {'family': ..., 'intent': ...} or None on failure.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    try:
        import urllib.request
        import urllib.error
    except ImportError:
        return None

    cand_lines = "\n".join(
        f"  {i+1}. {fam} (model confidence {prob:.2f})"
        for i, (fam, prob) in enumerate(top_candidates[:3])
    )
    valid_intents = ", ".join(_state["intents"])

    system_msg = (
        "You are a level design taxonomy expert. Pick the single best graph_family "
        "and intent for the user's level prompt. Respond in JSON only.")
    user_msg = (
        f"Prompt: {prompt!r}\n\n"
        f"Classifier top candidates:\n{cand_lines}\n\n"
        f"Valid intents: {valid_intents}\n\n"
        f'Reply with JSON: {{"family": "<one of the candidates>", "intent": "<one valid intent>"}}'
    )

    body = json.dumps({
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.1,
        "max_tokens": 100,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    try:
        text = data["choices"][0]["message"]["content"].strip()
        # Strip possible markdown fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        out = json.loads(text)
        fam = out.get("family")
        intent = out.get("intent")
        if fam not in _state["families"] or intent not in _state["intents"]:
            return None
        return {"family": fam, "intent": intent}
    except (KeyError, json.JSONDecodeError, IndexError):
        return None


# ─── Public API ──────────────────────────────────────────────────────

def is_available() -> bool:
    """Check if the classifier is loaded and ready."""
    return _load()


def get_load_error() -> Optional[str]:
    """Return the last load error message, if any."""
    _load()
    return _state["load_error"]


def classify_prompt(prompt: str, style: Optional[str] = None) -> Optional[dict]:
    """Classify a user prompt into (graph_family, intent).

    Args:
        prompt: Natural language description of the level.
        style:  Optional style key (e.g. 'medieval_keep'). If provided and
                in the training vocabulary, it is one-hot encoded as a side
                feature. If None, a zero vector is used (matches training-time
                style dropout).

    Returns:
        {
            "family": str,           # one of the trained graph_family classes
            "intent": str,           # one of the trained intent classes
            "confidence": float,     # softmax prob of the predicted family
            "top3": list[(str, float)],
            "intent_confidence": float,
            "deepseek_confirmed": bool,
        }
        Returns None if the classifier is unavailable.
    """
    if not _load():
        return None

    import torch

    model = _state["model"]
    encoder = _state["encoder"]
    families = _state["families"]
    intents = _state["intents"]
    style_to_idx = _state["style_to_idx"]
    style_dim = _state["style_dim"]

    with torch.no_grad():
        emb = encoder.encode([prompt], convert_to_numpy=True,
                             show_progress_bar=False)
        text_emb = torch.tensor(emb, dtype=torch.float32)

        style_vec = None
        if style_dim > 0:
            style_vec = torch.zeros(1, style_dim)
            if style and style in style_to_idx:
                style_vec[0, style_to_idx[style]] = 1.0

        out = model(text_emb, style_vec)
        fam_probs = torch.softmax(out["family"], dim=-1)[0]
        int_probs = torch.softmax(out["intent"], dim=-1)[0]

        fam_idx = int(fam_probs.argmax())
        int_idx = int(int_probs.argmax())
        confidence = float(fam_probs[fam_idx])
        intent_confidence = float(int_probs[int_idx])

        top3_idx = fam_probs.topk(min(3, len(families))).indices.tolist()
        top3 = [(families[i], float(fam_probs[i])) for i in top3_idx]

    family = families[fam_idx]
    intent = intents[int_idx]
    confirmed = False

    if confidence < CONFIDENCE_THRESHOLD:
        ds = _confirm_with_deepseek(prompt, top3, intent)
        if ds is not None:
            family = ds["family"]
            intent = ds["intent"]
            confirmed = True

    return {
        "family": family,
        "intent": intent,
        "confidence": confidence,
        "top3": top3,
        "intent_confidence": intent_confidence,
        "deepseek_confirmed": confirmed,
    }
