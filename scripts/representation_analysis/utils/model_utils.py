import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import warnings


def load_yolo_backbone(model_path: str, device: str = "cuda"):
    """
    Try to load model via ultralytics.YOLO, then find a backbone module and create an embedding extractor.
    Returns: dict with keys: model, extract_fn, preprocess, hook_handle, embedding_dim
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception:
        YOLO = None

    if YOLO:
        try:
            y = YOLO(model_path)
            y.model.to(device).eval()
            # freeze parameters
            for p in y.model.parameters():
                p.requires_grad = False
            # Heuristic: capture the last large feature map before head
            backbone_module = None
            # try known attribute names
            if hasattr(y.model, "model"):
                seq = list(y.model.model)
                # pick a layer near the end but before YOLO head (use -2 or -3)
                idx = max(0, len(seq) - 3)
                backbone_module = seq[idx]
            # fallback to entire model
            if backbone_module is None:
                backbone_module = y.model

            features = {}
            def hook_fn(module, inp, out):
                features['feat'] = out.detach()

            handle = backbone_module.register_forward_hook(hook_fn)

            def preprocess_pil(img_pil: Image.Image, img_size: int = 640):
                tf = T.Compose([
                    T.Resize((img_size, img_size)),
                    T.ToTensor(),
                ])
                return tf(img_pil).unsqueeze(0).to(device)

            def extract_embedding(img_tensor: torch.Tensor):
                features.clear()
                with torch.no_grad():
                    _ = y.model(img_tensor)
                    fmap = features.get('feat', None)
                    if fmap is None:
                        # fallback: use model output if any
                        raise RuntimeError("Could not capture backbone features from ultralytics model.")
                    # global average pooling over spatial dims
                    if fmap.ndim == 4:
                        emb = torch.mean(fmap, dim=[2,3])  # B x C
                    elif fmap.ndim == 3:
                        emb = torch.mean(fmap, dim=2)  # B x C
                    else:
                        emb = fmap.view(fmap.size(0), -1)
                    return emb.cpu().numpy()
            # try to infer embedding dim by running a dummy tensor
            dummy = torch.zeros(1,3,640,640).to(device)
            try:
                emb = extract_embedding(dummy)
                embedding_dim = emb.shape[1]
            except Exception:
                embedding_dim = 256
            # return with handle to remove hook later
            return {"model": y.model, "extract_fn": extract_embedding, "preprocess": preprocess_pil, "hook_handle": handle, "embedding_dim": embedding_dim}
        except Exception as e:
            warnings.warn(f"ultralytics YOLO load failed: {e}. Falling back to torch.load.")
    # Generic torch.load fallback (best-effort)
    ckpt = torch.load(model_path, map_location=device)
    model = None
    if isinstance(ckpt, dict) and "model" in ckpt:
        try:
            model = ckpt["model"]
        except Exception:
            pass
    if model is None:
        # user will need to supply a model architecture; here we try to load state_dict into a simple conv-net stub
        warnings.warn("Could not reconstruct YOLO model from checkpoint automatically. A best-effort stub will be used.")
        # Create a tiny stub backbone (not ideal, but keeps interface)
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    features = {}
    def hook_fn(module, inp, out):
        features['feat'] = out.detach()
    # pick last conv or last module
    last_mod = None
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Sequential):
            last_mod = m
            break
    if last_mod is None:
        last_mod = model
    handle = last_mod.register_forward_hook(hook_fn)
    def preprocess_pil(img_pil: Image.Image, img_size: int = 640):
        tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
        return tf(img_pil).unsqueeze(0).to(device)
    def extract_embedding(img_tensor: torch.Tensor):
        features.clear()
        with torch.no_grad():
            _ = model(img_tensor)
            fmap = features.get('feat', None)
            if fmap is None:
                raise RuntimeError("Could not capture features from fallback model.")
            if fmap.ndim == 4:
                emb = torch.mean(fmap, dim=[2,3])
            elif fmap.ndim == 3:
                emb = torch.mean(fmap, dim=2)
            else:
                emb = fmap.view(fmap.size(0), -1)
            return emb.cpu().numpy()
    # infer dim
    import torch as _torch
    dummy = _torch.zeros(1,3,640,640).to(device)
    try:
        embedding_dim = extract_embedding(dummy).shape[1]
    except Exception:
        embedding_dim = 256
    return {"model": model, "extract_fn": extract_embedding, "preprocess": preprocess_pil, "hook_handle": handle, "embedding_dim": embedding_dim}

