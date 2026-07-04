"""제출 zip 검증기 — 매 제출 전 필수 실행. (07-04 Line761 에러 재발 방지)
사용: python verify_zip.py <zip경로> [--smoke]
검사: ①CRC ②백슬래시 엔트리 ③script.py 루트 ④model/hf_model/config.json ⑤크기<1GB ⑥SHA256 출력
--smoke (노트북 WSL 전용): 클린룸 unzip → 오프라인 CPU 스모크까지."""
import sys, os, zipfile, hashlib

def fail(msg):
    print(f"  ❌ FAIL: {msg}")
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    path = sys.argv[1]
    smoke = "--smoke" in sys.argv
    if not os.path.exists(path):
        fail(f"파일 없음: {path}")
    size = os.path.getsize(path)
    print(f"검증: {path} ({size/1e6:.0f}MB)")

    # ⑤ 크기
    if size >= 1_000_000_000:
        fail(f"1GB 초과: {size}")
    print("  ✓ 크기 < 1GB")

    # ① CRC
    z = zipfile.ZipFile(path)
    bad = z.testzip()
    if bad:
        fail(f"CRC 손상: {bad}")
    print("  ✓ CRC 무결")

    names = z.namelist()
    # ② 백슬래시
    bs = [n for n in names if "\\" in n]
    if bs:
        fail(f"백슬래시 엔트리 {len(bs)}개 (Linux에서 model/ 못 읽음): {bs[:3]}")
    print(f"  ✓ 백슬래시 0 ({len(names)} entries)")

    # ③④ 필수 파일
    if "script.py" not in names:
        fail("script.py가 zip 루트에 없음")
    print("  ✓ script.py 루트")
    if not any(n == "model/hf_model/config.json" for n in names):
        fail("model/hf_model/config.json 없음 — ./model missing 에러 남")
    print("  ✓ model/hf_model/config.json")
    if "requirements.txt" not in names:
        print("  ⚠ requirements.txt 없음 (확인 필요)")

    # ⑥ SHA256 — 업로드 직전 파일과 대조용
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    print(f"  SHA256: {h.hexdigest()[:16]}...  ← 업로드한 파일과 이 값 대조")

    if smoke:
        import subprocess, shutil, tempfile
        print("=== 클린룸 스모크 (CPU offline) ===")
        w = tempfile.mkdtemp(prefix="zipsmoke_")
        with zipfile.ZipFile(path) as zz:
            zz.extractall(w)
        os.makedirs(os.path.join(w, "data"), exist_ok=True)
        for f in ["test.jsonl", "sample_submission.csv"]:
            for cand in ["/mnt/c/dacon/open/data", r"C:\dacon\open\data"]:
                s = os.path.join(cand, f)
                if os.path.exists(s):
                    shutil.copy(s, os.path.join(w, "data", f)); break
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        r = subprocess.run([sys.executable, "script.py"], cwd=w, env=env, capture_output=True, text=True, timeout=900)
        out = os.path.join(w, "output", "submission.csv")
        if r.returncode != 0 or not os.path.exists(out):
            print(r.stderr[-500:])
            fail("스모크 실패")
        lines = open(out, encoding="utf-8").read().splitlines()
        print(f"  ✓ 스모크 통과 rows={len(lines)-1}, sample={lines[1] if len(lines)>1 else ''}")
        shutil.rmtree(w, ignore_errors=True)

    print("✅ ALL PASSED — 제출 가능")

if __name__ == "__main__":
    main()
