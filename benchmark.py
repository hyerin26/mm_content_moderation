import time
import torch
import torchaudio
import pandas as pd
from PIL import Image
from torch.nn import functional as F

from transformers import (
    CLIPModel, CLIPProcessor,
    AutoImageProcessor, AutoModelForImageClassification,
    AutoProcessor, AutoModelForVision2Seq,
    AutoFeatureExtractor, AutoModelForAudioClassification,
    AutoModel, AutoTokenizer, AutoModelForSequenceClassification,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
precision = "FP16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "FP32"

# ===========================================================
# 공통 유틸
# ===========================================================

def measure_ms(fn):
    start = time.time()
    out = fn()
    end = time.time()
    return out, (end - start) * 1000.0  # ms

def get_model_stats(model):
    params = sum(p.numel() for p in model.parameters())
    # fp32 기준으로 size 계산 (정확한 메모리는 dtype에 따라 변함)
    size_mb = params * 4 / (1024 ** 2)
    return params, size_mb

results = []  # 최종 표용


# ===========================================================
# 1. IMAGE MODELS
#   - CLIP: 텍스트 프롬프트와 cosine similarity
#   - NSFW: normal vs nsfw 확률
#   - Florence-2: 텍스트 프롬프트 기반 VLM
# ===========================================================

def benchmark_clip(model_name, image_path, text_prompts):
    print(f"\n[IMAGE][CLIP] {model_name}")

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    params, size_mb = get_model_stats(model)

    image = Image.open(image_path).convert("RGB")

    # Preprocessing 측정
    (_, inputs_ms) = measure_ms(
        lambda: processor(
            text=text_prompts,
            images=image,
            return_tensors="pt"
        ).to(device)
    )
    inputs = processor(text=text_prompts, images=image, return_tensors="pt").to(device)

    # Warmup
    for _ in range(3):
        _ = model(**inputs)

    # Inference 측정
    (outputs, infer_ms) = measure_ms(lambda: model(**inputs))

    # CLIP: 이미지-텍스트 similarity
    logits_per_image = outputs.logits_per_image  # (batch=1, num_text)
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
    })


def benchmark_nsfw(model_name, image_path):
    print(f"\n[IMAGE][NSFW] {model_name}")

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)

    params, size_mb = get_model_stats(model)

    image = Image.open(image_path).convert("RGB")

    # Preprocessing
    (_, inputs_ms) = measure_ms(
        lambda: image_processor(images=image, return_tensors="pt").to(device)
    )
    inputs = image_processor(images=image, return_tensors="pt").to(device)

    # Warmup
    for _ in range(3):
        _ = model(**inputs)

    # Inference
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
    })


def benchmark_florence(model_name, image_path, prompt):
    print(f"\n[IMAGE][Florence-2] {model_name}")

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(model_name).to(device)

    params, size_mb = get_model_stats(model)

    image = Image.open(image_path).convert("RGB")

    # Preprocessing
    (_, inputs_ms) = measure_ms(
        lambda: processor(text=prompt, images=image, return_tensors="pt").to(device)
    )
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    # Warmup
    for _ in range(1):
        _ = model.generate(**inputs, max_new_tokens=32)

    # Inference (generate)
    (generated_ids, infer_ms) = measure_ms(
        lambda: model.generate(**inputs, max_new_tokens=32)
    )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"  - Prompt: {prompt}")
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
    })


# ===========================================================
# 2. AUDIO MODELS
#   - AST / PANNs: 527-class AudioSet 확률
#   - CLAP: audio-text similarity
# ===========================================================

def load_audio_waveform(audio_path, target_sr):
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.squeeze()
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform, target_sr


def benchmark_audio_classification(model_name, audio_path, arch_type):
    print(f"\n[AUDIO][CLASSIFICATION] {model_name} ({arch_type})")

    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name).to(device)

    params, size_mb = get_model_stats(model)

    waveform, sr = load_audio_waveform(audio_path, extractor.sampling_rate)

    # Preprocessing
    (_, inputs_ms) = measure_ms(
        lambda: extractor(waveform, sampling_rate=sr, return_tensors="pt").to(device)
    )
    inputs = extractor(waveform, sampling_rate=sr, return_tensors="pt").to(device)

    # Warmup
    for _ in range(3):
        _ = model(**inputs)

    # Inference
    (outputs, infer_ms) = measure_ms(lambda: model(**inputs))

    logits = outputs.logits
    probs = F.softmax(logits, dim=-1).detach().cpu()[0]
    id2label = model.config.id2label

    # top-5 label 출력
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
    })


def benchmark_clap(model_name, audio_path, candidate_labels):
    print(f"\n[AUDIO][CLAP] {model_name} (Dual-encoder)")

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    params, size_mb = get_model_stats(model)

    # Audio load
    dummy_sr = 48000  # CLAP processor가 내부적으로 맞춰줌
    waveform, _ = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0)  # mono

    # Preprocessing (audio + text 둘 다)
    (_, inputs_ms) = measure_ms(
        lambda: processor(
            audios=waveform,
            text=candidate_labels,
            return_tensors="pt",
            padding=True
        ).to(device)
    )
    inputs = processor(
        audios=waveform,
        text=candidate_labels,
        return_tensors="pt",
        padding=True
    ).to(device)

    # Warmup
    for _ in range(1):
        _ = model(**inputs)

    # Inference
    (outputs, infer_ms) = measure_ms(lambda: model(**inputs))

    # CLAP outputs: audio_embeds, text_embeds (모델에 따라 다를 수 있음)
    audio_embeds = outputs.audio_embeds  # (1, d)
    text_embeds = outputs.text_embeds    # (num_labels, d)

    audio_embeds = F.normalize(audio_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
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
    })


# ===========================================================
# 3. TEXT MODELS
#   - Toxic-BERT / Toxic-RoBERTa: toxicity scores
# ===========================================================

def benchmark_text_toxicity(model_name, text_path):
    print(f"\n[TEXT][TOXICITY] {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    params, size_mb = get_model_stats(model)

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Preprocessing
    (_, inputs_ms) = measure_ms(
        lambda: tokenizer(text, return_tensors="pt", truncation=True).to(device)
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)

    # Warmup
    for _ in range(3):
        _ = model(**inputs)

    # Inference
    (outputs, infer_ms) = measure_ms(lambda: model(**inputs))

    logits = outputs.logits.detach().cpu()[0]
    # multi-label일 수도, softmax single-label일 수도 있어서 둘 다 커버
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
    })


# ===========================================================
# MAIN
# ===========================================================

if __name__ == "__main__":
    harmful_image = "images/harmful.jpg"
    harmful_audio = "audio/harmful.wav"
    harmful_text = "text/harmful.txt"

    # 1. FRAME / IMAGE MODELS
    clip_prompts = ["a knife", "blood", "fighting", "a peaceful scene"]
    florence_prompt = "Describe any violence, weapons, or injuries in this image."

    benchmark_clip("openai/clip-vit-base-patch32", harmful_image, clip_prompts)
    benchmark_clip("openai/clip-vit-large-patch14", harmful_image, clip_prompts)

    benchmark_nsfw("Falconsai/nsfw_image_detection", harmful_image)

    benchmark_florence("microsoft/Florence-2-base", harmful_image, florence_prompt)
    benchmark_florence("microsoft/Florence-2-large", harmful_image, florence_prompt)

    # 2. AUDIO MODELS
    # AST (Transformer)
    benchmark_audio_classification(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        harmful_audio,
        arch_type="Transformer"
    )

    # PANNs (CNN) - 실제 허깅페이스 모델 이름은 환경에 맞게 조정 필요
    benchmark_audio_classification(
        "qiuqiangkong/panns_cnn14",
        harmful_audio,
        arch_type="CNN"
    )

    # CLAP (Dual-encoder)
    clap_labels = ["scream", "crying", "explosion", "gunshot", "music"]
    benchmark_clap(
        "laion/clap-htsat-unfused",
        harmful_audio,
        candidate_labels=clap_labels
    )

    # 3. TEXT MODELS
    benchmark_text_toxicity("unitary/unbiased-toxic-roberta", harmful_text)
    benchmark_text_toxicity("unitary/toxic-bert", harmful_text)

    # 4. 결과 표 출력
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
    ])

    print("\n\n========== BENCHMARK SUMMARY ==========\n")
    print(df.to_string(index=False))

    df.to_csv("benchmark_results.csv", index=False)
    print("\n결과가 benchmark_results.csv 로 저장되었습니다.")
