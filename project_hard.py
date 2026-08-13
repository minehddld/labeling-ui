import os, sys, time, json, re, torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

sys.stdout.reconfigure(line_buffering=True)  # 백그라운드/리다이렉트 실행 시에도 print가 바로바로 보이도록

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"   # 모델만 바꿔가며 재사용

# === 모델 간 성능 비교용 고정 설정 — 다른 모델로 바꿔서 재사용할 때도 절대 건드리지 말 것 ===
# (양자화 방식/토큰 한도가 모델마다 다르면 "모델 성능 차이"가 아니라 "설정 차이"를 비교하게 됨)
MAX_NEW_TOKENS = 768

TEST_SET = [
    ("hard1.jpg", "stud"),
    ("hard2.jpg", "stud"),
    ("hard3.jpg", "stud"),
    ("hard4.jpg", "stud"),
    ("hard5.jpg", "stud"),
    ("easy1.jpg", "stud"),
    ("easy2.jpg", "stud"),
    ("easy3.jpg", "stud"),
    ("easy4.jpg", "stud"),
    ("easy5.jpg", "stud")
]

def make_prompt(target):
    return (
        f'Detect every weld {target} (small raised pin, not the holes) in the image. '
        'Return only a JSON array: [{"bbox_2d":[x1,y1,x2,y2]}]. '
        'If none, return [].'
    )

def build_inputs(img, prompt):
    # Qwen 계열은 반드시 chat template을 거쳐야 이미지 placeholder 토큰이 삽입됨
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to("cuda")

def decode_new_tokens(inputs, out):
    # 입력으로 넣은 프롬프트 부분은 잘라내고, 새로 생성된 부분만 디코딩
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]

# 1. 모델 로드 (로드시간 · 메모리 측정)
# 4bit 양자화 — 8GB VRAM에서도 모든 비교 대상 모델을 동일 조건으로 돌리기 위해 전 모델 공통 적용.
# 모델마다 양자화 여부/방식을 다르게 하면 "모델 성능 차이"가 아니라 "양자화 차이"를 비교하게 됨.
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 고해상도 원본(2000~3000px대)을 그대로 먹이면 vision 토큰이 폭증해 이미지 1장에 수백 초가 걸림 —
# 상한을 걸어 내부적으로 리사이즈되게 함 (모델 간 비교용 고정값이므로 이후 모델 교체 시에도 유지)
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

print("[모델 로드] 시작...")
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, quantization_config=quant_config, device_map="cuda"
)
load_time = time.time() - t0
mem_after_load = torch.cuda.memory_allocated() / 1e9  # GB
print(f"[모델 로드] 완료 — {load_time:.1f}s, VRAM {mem_after_load:.1f}GB")

# 2. 워밍업 1회
print("[워밍업] 진행 중... (첫 추론은 커널 초기화 때문에 느릴 수 있음)")
_t_warm = time.time()
_img0 = Image.open(TEST_SET[0][0]).convert("RGB")
_inputs0 = build_inputs(_img0, make_prompt(TEST_SET[0][1]))
_ = model.generate(**_inputs0, max_new_tokens=MAX_NEW_TOKENS)
print(f"[워밍업] 완료 — {time.time() - _t_warm:.1f}s")

# 3. 본 측정
results = []
n_total = len(TEST_SET)
for idx, (img_path, target) in enumerate(TEST_SET, start=1):
    print(f"[{idx}/{n_total}] {img_path} ({target}) 추론 중...")
    img = Image.open(img_path).convert("RGB")
    prompt = make_prompt(target)

    t1 = time.time()
    inputs = build_inputs(img, prompt)
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    elapsed = time.time() - t1

    text = decode_new_tokens(inputs, out)

    try:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        boxes = json.loads(match.group())
        # 모델마다 좌표 키 이름이 다를 수 있어서 bbox/box/bbox_2d를 전부 bbox_2d로 통일
        for b in boxes:
            if "bbox_2d" not in b:
                for alt_key in ("bbox", "box", "box_2d"):
                    if alt_key in b:
                        b["bbox_2d"] = b.pop(alt_key)
                        break
        format_ok = isinstance(boxes, list) and all("bbox_2d" in b for b in boxes)
        n_detected = len(boxes)
    except Exception:
        format_ok = False
        n_detected = 0

    results.append({
        "image": img_path, "target": target, "time_s": round(elapsed, 2),
        "format_ok": format_ok, "n_detected": n_detected, "raw": text,
    })

    avg_time = sum(r["time_s"] for r in results) / idx
    remaining = n_total - idx
    eta = avg_time * remaining
    status = "OK" if format_ok else "FAIL"
    print(
        f"    -> {elapsed:.1f}s [{status}] {n_detected}개 검출  |  "
        f"평균 {avg_time:.1f}s/장, 남은 {remaining}장 예상 대기 {eta:.0f}s ({eta / 60:.1f}분)"
    )

# 4. 결과 출력
n_ok = sum(1 for r in results if r["format_ok"])
total_time = sum(r["time_s"] for r in results)

print()
print("=" * 60)
print("완료")
print("=" * 60)
print(f"모델           {MODEL_ID}")
print(f"설정           4bit nf4 양자화, max_new_tokens={MAX_NEW_TOKENS}  (모델 간 비교용 고정값)")
print(f"모델 로드      {load_time:.1f}s   VRAM {mem_after_load:.1f}GB")
print(f"총 추론 시간   {total_time:.1f}s ({total_time / 60:.1f}분)")
print(f"성공률         {n_ok}/{n_total}")
print("-" * 60)
for r in results:
    status = "OK  " if r["format_ok"] else "FAIL"
    print(f"[{status}] {r['image']:<12} {r['target']:<8} {r['time_s']:>7.1f}s  검출 {r['n_detected']}개")
print("-" * 60)

failed = [r for r in results if not r["format_ok"]]
if failed:
    print(f"실패한 이미지: {', '.join(r['image'] for r in failed)}")
    for r in failed:
        print(f"  [{r['image']}] raw 응답 끝부분: ...{r['raw'][-120:]}")
else:
    print("모든 이미지 JSON 파싱 성공")

# 5. 좌표 확인용 — 검출된 bbox를 원본 이미지 위에 그려서 저장 + 좌표 자체도 JSON으로 저장
# 모델별로 폴더를 분리 — 안 그러면 다른 모델로 바꿔 돌릴 때 이전 모델 결과가 덮어써짐
MODEL_SLUG = MODEL_ID.split("/")[-1]
OUT_DIR = os.path.join("boxed_output", MODEL_SLUG)
os.makedirs(OUT_DIR, exist_ok=True)

coords_export = []
for r in results:
    boxes = []
    img_w, img_h = Image.open(r["image"]).size
    if r["format_ok"]:
        boxes = json.loads(re.search(r"\[.*\]", r["raw"], re.DOTALL).group())
        img = Image.open(r["image"]).convert("RGB")
        draw = ImageDraw.Draw(img)
        for b in boxes:
            # Qwen이 반환하는 bbox_2d는 0~1000 정규화 좌표 — 원본 이미지 크기로 스케일링 필요
            x1, y1, x2, y2 = b["bbox_2d"]
            draw.rectangle(
                [x1 / 1000 * img_w, y1 / 1000 * img_h, x2 / 1000 * img_w, y2 / 1000 * img_h],
                outline="red", width=4,
            )
        img.save(os.path.join(OUT_DIR, r["image"]))

    coords_export.append({
        "image": r["image"], "target": r["target"], "time_s": r["time_s"],
        "format_ok": r["format_ok"], "n_detected": r["n_detected"],
        "image_size": [img_w, img_h], "coord_space": "normalized_0_1000",
        "boxes": boxes,
    })

export = {
    "model": MODEL_ID,
    "config": {"quantization": "4bit_nf4", "max_new_tokens": MAX_NEW_TOKENS},
    "load_time_s": round(load_time, 2),
    "mem_after_load_gb": round(mem_after_load, 2),
    "n_ok": n_ok,
    "n_total": n_total,
    "results": coords_export,
}
with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
    json.dump(export, f, ensure_ascii=False, indent=2)

print(f"좌표 확인용 이미지 {n_ok}장 + results.json 저장 완료 -> {OUT_DIR}/ 폴더")