# Privacy (SoM + DualTAP)

## Current behavior

- **Screenshots (SoM / vision agents):** optional **DualTAP** adversarial perturbation when
  `privacy_backend: dualtap` (or `PRIVACY_BACKEND=dualtap`) and `dualtap_checkpoint` / `DUALTAP_CHECKPOINT`
  is set. Implemented in `utils_mobile/privacy/dualtap_adapter.py`, applied in
  `recorder/json_recoder.py` (`update_before`).

- **Prompts and XML:** no GLiNER / token anonymization. Instructions and compressed UI text are sent
  as-is to the model.

- **`PrivacyProtectionLayer`** (`utils_mobile/privacy/layer.py`): minimal stub — disabled by default
  (`enabled=False`). Keeps optional token registry and `cloud_agent_compute_with_tokens` for advanced
  use; statistics / token files from old GLiNER runs can still be aggregated if present.

## Defaults

- `PRIVACY_BACKEND` default is `none` (no DualTAP). Set to `dualtap` and configure checkpoint for
  image-only protection.

## Removed

- GLiNER + regex PII pipeline, EasyOCR screenshot masking, and XML/prompt masking via
  `identify_and_mask_*` / `anonymize_prompt` (files `detection.py`, `screenshot.py`, `string_utils.py`
  removed).
