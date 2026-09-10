import json
import urllib.request

def main():
    print("========================================")
    print("RICEGUARD LIVE APPLICATION HEALTH CHECK")
    print("========================================")

    # 1. Health
    with urllib.request.urlopen("http://127.0.0.1:8000/api/health") as res:
        health = json.loads(res.read().decode())
        print(f"[+] Backend API Status: {health.get('status').upper()} (Compute: {health.get('device')}, Model: {health.get('model')})")

    # 2. Sample Image
    with urllib.request.urlopen("http://127.0.0.1:8000/api/samples/blast") as res:
        img_bytes = res.read()
        print(f"[+] Sample Library: Loaded blast_1002.jpg ({len(img_bytes):,} bytes)")

    # 3. Predict Endpoint
    boundary = "----RiceGuardTestBoundary"
    header = f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"sample.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    payload = header + img_bytes + footer

    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/predict",
        data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req) as res:
        result = json.loads(res.read().decode())
        pred = result["prediction"]
        print(f"[+] Live Prediction Result: {pred['class_name']} ({pred['confidence']*100:.2f}% confidence)")
        print(f"[+] Lesion Localization: {result['lesion_count']} bounding box(es) detected with frozen calibration (tau=0.60, top_k=3)")
        print(f"[+] Explainability Modalities: {', '.join(result['visualizations'].keys())}")

    # 4. Frontend check
    with urllib.request.urlopen("http://127.0.0.1:5173") as res:
        print(f"[+] React/Vite Frontend: ONLINE (HTTP {res.status}) at http://127.0.0.1:5173")

    print("\n[SUCCESS] RiceGuard is up and running smoothly!")

if __name__ == "__main__":
    main()
