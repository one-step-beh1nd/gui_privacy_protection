import numpy as np
import torch
from PIL import Image

from .config import Config
from .generator import NoiseGenerator


def _load_rgb_image(image_path):
    with Image.open(image_path) as image:
        return image.convert("RGB")


def _pil_to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    return torch.from_numpy(array).permute(2, 0, 1)


def _tensor_to_pil(image_tensor):
    array = image_tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    array = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def load_generator(checkpoint_path, config):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_cfg = checkpoint.get("config", {}) or {}
    gen_kwargs = dict(
        in_channels=3,
        out_channels=3,
        epsilon=saved_cfg.get("epsilon", config.epsilon),
        attn_gamma=saved_cfg.get("attn_gamma", 1.0),
        attn_threshold=saved_cfg.get("attn_threshold", 0.0),
        attn_topk_percent=saved_cfg.get("attn_topk_percent", 0.0),
        attn_mix=saved_cfg.get("attn_mix", 1.0),
        attn_dilate_kernel=saved_cfg.get("attn_dilate_kernel", 1),
        attn_renorm=saved_cfg.get("attn_renorm", False),
        attn_as_epsilon=saved_cfg.get("attn_as_epsilon", False),
        attn_integration=saved_cfg.get("attn_integration", "film"),
        film_hidden=saved_cfg.get("film_hidden", 32),
        film_strength=saved_cfg.get("film_strength", 1.0),
    )

    generator = NoiseGenerator(**gen_kwargs).to(device)
    state = checkpoint["generator_state_dict"]

    if any(key.endswith(".up.weight") for key in state.keys()):
        new_state = state.copy()
        for name in ["up1", "up2", "up3", "up4"]:
            weight_key = f"{name}.up.weight"
            bias_key = f"{name}.up.bias"
            if weight_key in state:
                weights = state[weight_key]
                reduced = weights.mean(dim=(2, 3)).permute(1, 0).unsqueeze(-1).unsqueeze(-1)
                new_state[f"{name}.reduce.weight"] = reduced
                if bias_key in state:
                    new_state[f"{name}.reduce.bias"] = state[bias_key]
                new_state.pop(weight_key, None)
                new_state.pop(bias_key, None)
        generator.load_state_dict(new_state, strict=False)
    else:
        try:
            generator.load_state_dict(state)
        except RuntimeError:
            generator.load_state_dict(state, strict=False)

    generator.eval()

    for key in [
        "surrogate_model_name",
        "attn_method",
        "use_attention",
        "image_size",
        "attn_gamma",
        "attn_threshold",
        "attn_topk_percent",
        "attn_mix",
        "attn_dilate_kernel",
        "attn_renorm",
        "attn_as_epsilon",
        "attn_integration",
        "film_hidden",
        "film_strength",
    ]:
        if key in saved_cfg:
            try:
                setattr(config, key, saved_cfg[key])
            except Exception:
                pass

    return generator, device


def generate_adversarial_image(image_path, generator, device, image_size=448, attention_map=None):
    original_image = _load_rgb_image(image_path)
    resized_image = original_image.resize((image_size, image_size), Image.BILINEAR)
    image_tensor = _pil_to_tensor(resized_image).unsqueeze(0).to(device)

    with torch.no_grad():
        use_attention = False
        if attention_map is not None:
            try:
                mask_probe = generator.shape_attention_map(
                    attention_map,
                    target_size=image_tensor.shape[-2:],
                    out_channels=1,
                )
                use_attention = bool(float(mask_probe.max().item()) > 1e-6)
            except Exception:
                use_attention = False

        if use_attention:
            delta = generator(image_tensor, attention_map=attention_map)
            adversarial_tensor = (image_tensor + delta).clamp(0.0, 1.0)
            noise = delta
        else:
            noise = generator(image_tensor)
            adversarial_tensor = generator.generate_adversarial(image_tensor)

    original_image_resized = _tensor_to_pil(image_tensor.squeeze(0))
    adversarial_image = _tensor_to_pil(adversarial_tensor.squeeze(0))

    return original_image_resized, adversarial_image, image_tensor, adversarial_tensor, noise


__all__ = ["Config", "generate_adversarial_image", "load_generator"]
