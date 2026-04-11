class Config:
    data_root = "./data"
    image_size = 448
    train_split_ratio = 0.8

    epsilon = 128.0 / 255.0

    batch_size = 4
    num_epochs = 10
    learning_rate = 1e-4

    alpha = 1.0
    beta = 1.0

    surrogate_model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    checkpoint_dir = "./checkpoints"
    save_interval = 60

    use_attention = True
    save_attention = True
    attention_dir = "./logs_eot/attn"
    attn_method = "contrast_pixel_grad"
    attn_gamma = 4.0
    attn_threshold = 0.85
    attn_topk_percent = 30
    attn_mix = 0.95
    attn_dilate_kernel = 3
    attn_renorm = True
    attn_integration = "film"
    film_hidden = 32
    film_strength = 1.0
    attn_as_epsilon = False
    device = "cuda"
    log_dir = "./logs"
    eval_interval = 1
    vis_interval = 10
    vis_noise_only = False

    test_single_app = "real"
