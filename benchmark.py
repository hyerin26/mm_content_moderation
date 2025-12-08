import time
import torch
import torchaudio
import pandas as pd
from PIL import Image
from torch.nn import functional as F

import os
os.environ["PYTORCH_SDPA_DISABLE"] = "1"

from transformers import (
    CLIPModel, CLIPProcessor,
    AutoImageProcessor, AutoModelForImageClassification,
    AutoProcessor, AutoModelForVision2Seq,
    AutoFeatureExtractor, AutoModelForAudioClassification,
    AutoModel, AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForCausalLM
)

device = "cuda" if torch.cuda.is_available() else "cpu"
precision = "FP16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "FP32"


# ===========================================================
#  공통 유틸
# ===========================================================

def measure_ms(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    return out, (end - start) * 1000.0


def get_model_stats(model):
    params = sum(p.numel() for p in model.parameters())
    size_mb = params * 4 / (1024 ** 2)
    return params, size_mb


results = []


# ===========================================================
# 1. IMAGE MODELS
# ===========================================================

def benchmark_clip(model_name, image_path, text_prompts, sample_type="harmful"):
    print(f"\n[IMAGE][CLIP] {model_name}")

    model = CLIPModel.from_pretrained(
        model_name,
        use_safetensors=True,
        revision="main",
        trust_remote_code=True
    ).to(device)

    processor = CLIPProcessor.from_pretrained(model_name)

    params, size_mb = get_model_stats(model)
    image = Image.open(image_path).convert("RGB")

    # Preprocessing
    (_, inputs_ms) = measure_ms(
        lambda: processor(
            text=text_prompts,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
    )
    inputs = processor(
        text=text_prompts,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    # Warmup
    for _ in range(3):
        _ = model(**inputs)

    # Inference
    (outputs, infer_ms) = measure_ms(lambda: model(**inputs))

    # 수정 사항: detach() 추가
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()[0]

    print("  - Text prompts & probs:")
    for p, s in zip(text_prompts, probs):
        print(f"    '{p}': {s:.4f}")
    top_idx = probs.argmax()
    print(f"  -> Predicted: '{text_prompts[top_idx]}'")

    total_ms = inputs_ms + infer_ms
    results.append({
        "model_name": model_name,
        "architecture_type": "Transformer",
        "parameter_count": params,
        "model_size_MB": size_mb,
        "preprocessing_cost_ms": inputs_ms,
        "inference_latency_ms": infer_ms,
        "total_latency_ms": total_ms,
        "device": device,
        "precision": precision,
        "modality": "image",
        "accuracy": "N/A",
        "sample_type": sample_type
    })


def benchmark_nsfw(model_name, image_path, sample_type="harmful"):
    print(f"\n[IMAGE][NSFW] {model_name}")

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)

    params, size_mb = get_model_stats(model)
    image = Image.open(image_path).convert("RGB")

    (_, inputs_ms) = measure_ms(
        lambda: image_processor(images=image, return_tensors="pt").to(device)
    )
    inputs = image_processor(images=image, return_tensors="pt").to(device)

    for _ in range(3):
        _ = model(**inputs)

    (outputs, infer_ms) = measure_ms(lambda: model(**inputs))

    logits = outputs.logits
    probs = F.softmax(logits, dim=-1).detach().cpu()[0]

    id2label = model.config.id2label
    print("  - Class probs:")
    for idx, prob in enumerate(probs):
        print(f"    {id2label[idx]}: {prob:.4f}")

    nsfw_label = probs.argmax().item()
    print(f"  -> Predicted: {id2label[nsfw_label]}")

    total_ms = inputs_ms + infer_ms
    results.append({
        "model_name": model_name,
        "architecture_type": "CNN",
        "parameter_count": params,
        "model_size_MB": size_mb,
        "preprocessing_cost_ms": inputs_ms,
        "inference_latency_ms": infer_ms,
        "total_latency_ms": total_ms,
        "device": device,
        "precision": precision,
        "modality": "image",
        "accuracy": "N/A",
        "sample_type": sample_type
    })


def benchmark_florence(model_name, image_path, prompt, sample_type="harmful"):
    print(f"\n[IMAGE][Florence-2] {model_name}")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    ).to(device)

    params, size_mb = get_model_stats(model)
    image = Image.open(image_path).convert("RGB")

    (_, inputs_ms) = measure_ms(lambda: processor(text=prompt, images=image, return_tensors="pt"))
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    inputs["pixel_values"] = inputs["pixel_values"].to(device, dtype=torch.float16)
    inputs["input_ids"] = inputs["input_ids"].to(device)
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].to(device)

    for _ in range(1):
        _ = model.generate(**inputs, max_new_tokens=32)

    (_, infer_ms) = measure_ms(lambda: model.generate(**inputs, max_new_tokens=32))

    text = processor.batch_decode(_, skip_special_tokens=True)[0]
    print(f"  -> Generated: {text}")

    total_ms = inputs_ms + infer_ms
    results.append({
        "model_name": model_name,
        "architecture_type": "Vision-Language",
        "parameter_count": params,
        "model_size_MB": size_mb,
        "preprocessing_cost_ms": inputs_ms,
        "inference_latency_ms": infer_ms,
        "total_latency_ms": total_ms,
        "device": device,
        "precision": precision,
        "modality": "image",
        "accuracy": "N/A",
        "sample_type": sample_type
    })


# ===========================================================
# 2. AUDIO MODELS
# ===========================================================

def load_audio_fixed(path, target_sr=16000, min_len=400):
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        sr = target_sr
    if waveform.shape[1] < min_len:
        waveform = torch.nn.functional.pad(waveform, (0, min_len - waveform.shape[1]))
    return waveform, sr


def load_audio_waveform(path, target_sr):
    return load_audio_fixed(path, target_sr)


def benchmark_audio_classification(model_name, audio_path, arch_type, sample_type="harmful"):
    print(f"\n[AUDIO][CLASSIFICATION] {model_name} ({arch_type})")

    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name).to(device)

    params, size_mb = get_model_stats(model)
    waveform, sr = load_audio_waveform(audio_path, extractor.sampling_rate)
    waveform = waveform.squeeze(0)

    (_, inputs_ms) = measure_ms(lambda: extractor(waveform, sampling_rate=sr, return_tensors="pt").to(device))
    inputs = extractor(waveform, sampling_rate=sr, return_tensors="pt").to(device)

    for _ in range(3):
        _ = model(**inputs)

    (outputs, infer_ms) = measure_ms(lambda: model(**inputs))

    logits = outputs.logits
    probs = F.softmax(logits, dim=-1).detach().cpu()[0]
    id2label = model.config.id2label

    topk = torch.topk(probs, k=min(5, probs.shape[0]))
    print("  - Top classes:")
    for score, idx in zip(topk.values, topk.indices):
        print(f"    {id2label[idx.item()]}: {score.item():.4f}")

    total_ms = inputs_ms + infer_ms
    results.append({
        "model_name": model_name,
        "architecture_type": arch_type,
        "parameter_count": params,
        "model_size_MB": size_mb,
        "preprocessing_cost_ms": inputs_ms,
        "inference_latency_ms": infer_ms,
        "total_latency_ms": total_ms,
        "device": device,
        "precision": precision,
        "modality": "audio",
        "accuracy": "N/A",
        "sample_type": sample_type
    })


def benchmark_clap(model_name, audio_path, candidate_labels, sample_type="harmful"):
    print(f"\n[AUDIO][CLAP] {model_name} (Dual-encoder)")

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    params, size_mb = get_model_stats(model)

    waveform, _ = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0)

    (_, inputs_ms) = measure_ms(
        lambda: processor(audios=waveform, text=candidate_labels,
                          return_tensors="pt", padding=True).to(device)
    )
    inputs = processor(audios=waveform, text=candidate_labels,
                       return_tensors="pt", padding=True).to(device)

    for _ in range(1):
        _ = model(**inputs)

    (outputs, infer_ms) = measure_ms(lambda: model(**inputs))

    audio_embeds = F.normalize(outputs.audio_embeds, dim=-1)
    text_embeds = F.normalize(outputs.text_embeds, dim=-1)
    sims = (audio_embeds @ text_embeds.T).squeeze(0).detach().cpu()

    print("  - Candidate label similarities:")
    for lbl, s in zip(candidate_labels, sims):
        print(f"    {lbl}: {s.item():.4f}")

    top_idx = torch.argmax(sims).item()
    print(f"  -> Predicted label: {candidate_labels[top_idx]}")

    total_ms = inputs_ms + infer_ms
    results.append({
        "model_name": model_name,
        "architecture_type": "Dual-encoder",
        "parameter_count": params,
        "model_size_MB": size_mb,
        "preprocessing_cost_ms": inputs_ms,
        "inference_latency_ms": infer_ms,
        "total_latency_ms": total_ms,
        "device": device,
        "precision": precision,
        "modality": "audio",
        "accuracy": "N/A",
        "sample_type": sample_type
    })


from panns_inference import AudioTagging

def benchmark_audio_panns(model_name, audio_path, sample_type="harmful"):
    print(f"\n[AUDIO][PANNs] {model_name} (CNN14)")

    (_, load_ms) = measure_ms(lambda: AudioTagging(checkpoint_path=None))
    at_model = AudioTagging(checkpoint_path=None)

    params = 79_000_000
    size_mb = 300

    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    wav_np = waveform.squeeze(0).cpu().numpy()
    if wav_np.ndim == 1:
        wav_np = wav_np[None, :]

    (out, infer_ms) = measure_ms(lambda: at_model.inference(wav_np))

    clipwise_output, embedding = out
    clipwise_output = clipwise_output.squeeze()
    labels = at_model.labels

    top5_idx = clipwise_output.argsort()[-5:][::-1]

    print("  - Top classes:")
    for idx in top5_idx:
        print(f"    {labels[idx]}: {clipwise_output[idx]:.4f}")

    total_ms = load_ms + infer_ms

    results.append({
        "model_name": model_name,
        "architecture_type": "CNN (PANNs-Cnn14)",
        "parameter_count": params,
        "model_size_MB": size_mb,
        "preprocessing_cost_ms": load_ms,
        "inference_latency_ms": infer_ms,
        "total_latency_ms": total_ms,
        "device": device,
        "precision": "FP32",
        "modality": "audio",
        "accuracy": "N/A",
        "sample_type": sample_type
    })


# ===========================================================
# 3. TEXT MODELS
# ===========================================================

def benchmark_text_toxicity(model_name, text_path, sample_type="harmful"):
    print(f"\n[TEXT][TOXICITY] {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    params, size_mb = get_model_stats(model)

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    (_, inputs_ms) = measure_ms(
        lambda: tokenizer(text, return_tensors="pt", truncation=True).to(device)
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)

    for _ in range(3):
        _ = model(**inputs)

    (outputs, infer_ms) = measure_ms(lambda: model(**inputs))

    logits = outputs.logits.detach().cpu()[0]
    if model.config.problem_type == "multi_label_classification":
        probs = torch.sigmoid(logits)
    else:
        probs = F.softmax(logits, dim=-1)

    id2label = model.config.id2label
    topk = torch.topk(probs, k=min(5, probs.shape[0]))

    print("  - Toxicity scores:")
    for score, idx in zip(topk.values, topk.indices):
        print(f"    {id2label[idx.item()]}: {score.item():.4f}")

    total_ms = inputs_ms + infer_ms

    results.append({
        "model_name": model_name,
        "architecture_type": "Transformer",
        "parameter_count": params,
        "model_size_MB": size_mb,
        "preprocessing_cost_ms": inputs_ms,
        "inference_latency_ms": infer_ms,
        "total_latency_ms": total_ms,
        "device": device,
        "precision": precision,
        "modality": "text",
        "accuracy": "N/A",
        "sample_type": sample_type
    })


# ===========================================================
# MAIN LOOP
# ===========================================================

if __name__ == "__main__":

    datasets = {
        "harmful": {
            "image": "images/harmful.jpg",
            "audio": "audio/harmful.wav",
            "text":  "text/harmful.txt",
        },
        "normal": {
            "image": "images/normal.jpg",
            "audio": "audio/normal.wav",
            "text":  "text/normal.txt",
        }
    }

    clip_prompts = ["a knife", "blood", "fighting", "a peaceful scene"]
    clap_labels = ["scream", "crying", "explosion", "gunshot", "music", "kitten"]
    florence_prompt = "Describe any violence, weapons, or injuries in this image."

    for sample_type, paths in datasets.items():

        print(f"\n\n========== Running benchmarks for {sample_type.upper()} data ==========\n")

        img_path = paths["image"]
        aud_path = paths["audio"]
        txt_path = paths["text"]

        benchmark_clip("openai/clip-vit-base-patch32", img_path, clip_prompts, sample_type)
        benchmark_clip("openai/clip-vit-large-patch14", img_path, clip_prompts, sample_type)
        benchmark_nsfw("Falconsai/nsfw_image_detection", img_path, sample_type)

        benchmark_audio_classification(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            aud_path,
            arch_type="Transformer",
            sample_type=sample_type
        )

        benchmark_audio_panns("PANNs-Cnn14", aud_path, sample_type)

        benchmark_clap(
            "laion/clap-htsat-unfused",
            aud_path,
            candidate_labels=clap_labels,
            sample_type=sample_type
        )

        benchmark_text_toxicity("unitary/unbiased-toxic-roberta", txt_path, sample_type)
        benchmark_text_toxicity("unitary/toxic-bert", txt_path, sample_type)

    df = pd.DataFrame(results, columns=[
        "model_name",
        "architecture_type",
        "parameter_count",
        "model_size_MB",
        "preprocessing_cost_ms",
        "inference_latency_ms",
        "total_latency_ms",
        "device",
        "precision",
        "modality",
        "accuracy",
        "sample_type"
    ])

    print("\n\n========== BENCHMARK SUMMARY ==========\n")
    print(df.to_string(index=False))

    df.to_csv("benchmark_results.csv", index=False)
    print("\n결과가 benchmark_results.csv 로 저장되었습니다.")
