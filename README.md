# Robot-and-VLM

یک ریپوی واحد برای **دموی ویلچر (Three.js)** + **پل زندهٔ VLA** (`live_frame_server`): فریم دوربین → Grounding DINO + SAM → تصویر overlay، و حرکت گسسته روی مسیر در `Robot/index.html`.

## ساختار

| مسیر | نقش |
|------|-----|
| `Robot/` | صفحهٔ وب، صحنهٔ ۳بعدی، مسیر‌یابی، اتصال به API |
| `VLA/` | کد پروژهٔ VLA، اسکریپت `scripts/live_frame_server.py`، کانفیگ `config/live_robot_bridge.yaml` |
| `start_stack.sh` | بالا آوردن همزمان سرور استاتیک Robot و API روی پورت‌های پیش‌فرض |

## پیش‌نیاز

- **Python 3.10+** (مثل سیستم شما)
- **GPU + PyTorch با CUDA** برای DINO/SAM واقعی (بعد از `pip install` طبق `VLA/README.md` در صورت نیاز نسخهٔ `torch` مناسب نصب کنید)
- مدل‌ها داخل ریپو نیستند؛ بعد از کلون باید دانلود شوند (پوشهٔ `VLA/models/` در `.gitignore` است).

## راه‌اندازی سریع

```bash
cd VLA
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_models.py --root models
cd ..
chmod +x start_stack.sh
./start_stack.sh
```

- **رابط دمو (مرورگر):** http://127.0.0.1:8765/ — اینجا خانه و ویلچر را می‌بینی.
- **API پردازش فریم:** http://127.0.0.1:8787 — فقط برای POST `/process_frame` و `/health`؛ اگر مستقیم در تب بازش کنی، صفحهٔ راهنما یا JSON می‌بینی، نه صحنهٔ ۳بعدی.
- در HUD دمو، فیلد **VLA API** باید مثلاً `http://127.0.0.1:8787` باشد.

متغیرهای مفید (اختیاری): `VLA_NUM_THREADS`, `ROBOT_PORT`, `VLA_PORT`, `ROBOT_PUBLIC_URL` (برای لینک روی صفحهٔ روت API).

## توسعه

- کانفیگ لایو: `VLA/config/live_robot_bridge.yaml`
- سرور: `VLA/scripts/live_frame_server.py`
