import runpod
import torch
import base64
import os
from pathlib import Path
from io import BytesIO
from PIL import Image

from flux2.util import load_flow_model, load_ae, load_text_encoder
from flux2.sampling import denoise, get_schedule, unpack

MODEL_PATH = os.environ.get("KLEIN_4B_MODEL_PATH", "/runpod-volume/models/FLUX.2-klein-4B")

print("Ucitavam model...")
device = torch.device("cuda")

model = load_flow_model("klein-4b", device=device)
ae = load_ae("klein-4b", device=device)
text_encoder = load_text_encoder("klein-4b", device=device)
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

    with torch.inference_mode():
        result = denoise(
            model=model,
            ae=ae,
            text_encoder=text_encoder,
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            input_images=input_images,
            generator=rng,
            device=device,
        )

    return {"image": image_to_b64(result), "seed": seed}


runpod.serverless.start({"handler": handler})
