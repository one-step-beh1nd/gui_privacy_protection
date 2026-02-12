"""
Utilities for PrivScreen evaluation (e.g. compute_text_metrics).
"""

import os
import time
import torch
import numpy as np
from typing import Dict, Optional

_bert_scorer = None
_st_model = None
_rouge = None
_sacrebleu = None
_psutil = None


def _lazy_import_bert_scorer():
    global _bert_scorer
    if _bert_scorer is None:
        try:
            from bert_score import BERTScorer
            _bert_scorer = BERTScorer(lang="en", model_type="microsoft/deberta-base-mnli", rescale_with_baseline=True)
        except Exception:
            _bert_scorer = False
    return _bert_scorer


def _lazy_import_st_model():
    global _st_model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            _st_model = False
    return _st_model


def _lazy_import_rouge():
    global _rouge
    if _rouge is None:
        try:
            from rouge_score import rouge_scorer
            _rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        except Exception:
            _rouge = False
    return _rouge


def _lazy_import_sacrebleu():
    global _sacrebleu
    if _sacrebleu is None:
        try:
            import sacrebleu
            _sacrebleu = sacrebleu
        except Exception:
            _sacrebleu = False
    return _sacrebleu


def compute_text_metrics(pred_text: str, true_text: str) -> Dict[str, Optional[float]]:
    pred = (pred_text or "").strip()
    ref = (true_text or "").strip()
    if not pred or not ref:
        return {
            "bertscore_f1": 0.0 if pred or ref else None,
            "cosine_sim": 0.0 if pred or ref else None,
            "bleu": 0.0 if pred or ref else None,
            "rouge_l": 0.0 if pred or ref else None,
        }
    bert_f1 = None
    scorer = _lazy_import_bert_scorer()
    if scorer is not False:
        try:
            P, R, F1 = scorer.score([pred], [ref])
            bert_f1 = float(F1.mean().item())
        except Exception:
            pass
    cosine_sim = None
    st = _lazy_import_st_model()
    if st is not False:
        try:
            import numpy as _np
            pred_emb = st.encode([pred], normalize_embeddings=True)
            ref_emb = st.encode([ref], normalize_embeddings=True)
            cosine_sim = float((_np.asarray(pred_emb) @ _np.asarray(ref_emb).T).item())
        except Exception:
            pass
    bleu = None
    sb = _lazy_import_sacrebleu()
    if sb is not False:
        try:
            bleu = float(sb.sentence_bleu(pred, [ref]).score / 100.0)
        except Exception:
            pass
    rouge_l = None
    rs = _lazy_import_rouge()
    if rs is not False:
        try:
            scores = rs.score(ref, pred)
            rouge_l = float(scores["rougeL"].fmeasure)
        except Exception:
            pass
    return {
        "bertscore_f1": bert_f1,
        "cosine_sim": cosine_sim,
        "bleu": bleu,
        "rouge_l": rouge_l,
    }
