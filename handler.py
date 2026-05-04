import runpod
import torch
import base64
import os
from io import BytesIO
from PIL import Image
from einops import rearrange

from flux2.util import load_flow_model, load_ae, load_text_encoder
from flux2.sampling import (
    denoise,
    get_schedule,
    encode_image_refs,
    batched_prc_txt,
)

MODEL_VARIANT = "flux.2-klein-4b"
os.environ["KLEIN_4B_MODEL_PATH"] = "/runpod-volume/models/FLUX.2-klein-4B/flux-2-klein-4b.safetensors"
os.environ["AE_MODEL_PATH"] = "/runpod-volume/models/FLUX.2-klein-4B/ae.safetensors"

print("Ucitavam model...")
device = torch.device("cuda")

model = load_flow_model(MODEL_VARIANT, device=device)
ae = load_ae(MODEL_VARIANT, device=device)
text_encoder = load_text_encoder(MODEL_VARIANT, device=device)
print("Model ucitan!")


def b64_to_image(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


def image_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def handler(job):
    inp = job["input"]

    prompt = inp.get("prompt", "Make this person wear these clothes. Keep the person identical, only change the clothing.")
    width = inp.get("width", 1024)
    height = inp.get("height", 1024)
    num_steps = inp.get("num_steps", 25)
    guidance = inp.get("guidance", 4.0)
    seed = inp.get("seed", None)

    input_images = []
    if "person_image" in inp:
        input_images.append(b64_to_image(inp["person_image"]))
    for img_b64 in inp.get("clothing_images", []):
        input_images.append(b64_to_image(img_b64))

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
    else:
        seed = int(rng.seed())

    with torch.inference_mode():
        # Encode text
        txt_tokens = text_encoder([prompt])
        txt, txt_ids = batched_prc_txt(txt_tokens)
        txt = txt.to(device, dtype=torch.bfloat16)
        txt_ids = txt_ids.to(device)

        # Prepare noise
        h_lat = height // 16
        w_lat = width // 16
        img_noise = torch.randn(
            1, h_lat * w_lat, 64,
            device=device, dtype=torch.bfloat16, generator=rng
        )
        img_ids = torch.zeros(1, h_lat * w_lat, 4, device=device, dtype=torch.int32)
        for i in range(h_lat):
            for j in range(w_lat):
                img_ids[0, i * w_lat + j, 1] = i
                img_ids[0, i * w_lat + j, 2] = j

        # Encode reference images
        img_cond_seq, img_cond_seq_ids = None, None
        if input_images:
            img_cond_seq, img_cond_seq_ids = encode_image_refs(ae, input_images)
            img_cond_seq = img_cond_seq.to(device)
            img_cond_seq_ids = img_cond_seq_ids.to(device)

        # Denoise
        timesteps = get_schedule(num_steps, h_lat * w_lat)
        result_lat = denoise(
            model=model,
            img=img_noise,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            guidance=guidance,
            img_cond_seq=img_cond_seq,
            img_cond_seq_ids=img_cond_seq_ids,
        )

        # Decode
        result_lat = rearrange(result_lat, "b (h w) c -> b c h w", h=h_lat, w=w_lat)
        result_img = ae.decode(result_lat)
        result_img = ((result_img + 1) / 2).clamp(0, 1)
        result_pil = Image.fromarray(
            (result_img[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype("uint8")
        )

    return {"image": image_to_b64(result_pil), "seed": seed}


runpod.serverless.start({"handler": handler})
