# Wheelchair path — 3D home demo

**خلاصه:** یک صفحهٔ تکی (`index.html`) با Three.js که خانهٔ چهار فضایی (نشیمن، آشپزخانه، راهرو، اتاق خواب)، مسیر‌یابی ویلچر روی گراف درها + شبکه، دو دوربین (نمای اصلی + دوربین ویلچر)، و **زیرساخت تشخیص مانع از روی دوربین ویلچر** (پرتوهای NDC شبیه سنسور عمق) را نشان می‌دهد. هدف مرحلهٔ بعد: با خواندن فاصله تا آبجکت‌ها، قبل از برخورد **مسیر را عوض کنید** — API و قلاب `onNearObstacle` برای این کار آماده است.

---

## مخزن GitHub

ریپو: **[rezabahadori76/Robot](https://github.com/rezabahadori76/Robot)** — ریشهٔ `main` همان فایل‌های این اپ است (بدون زیرپوشهٔ اضافه).

```bash
git clone https://github.com/rezabahadori76/Robot.git
cd Robot
python3 -m http.server 8080
# سپس در مرورگر: http://localhost:8080/index.html
```

### پوش کردن تغییرات (روی سیستم خودت)

از پوشه‌ای که `.git` دارد (همین پروژهٔ لوکال):

```bash
git status
git remote -v   # انتظار: origin → https://github.com/rezabahadori76/Robot.git
git push -u origin main
```

- با **HTTPS** اگر رمز خواست: از [GitHub PAT](https://github.com/settings/tokens) به‌جای پسورد استفاده کن، یا `gh auth login` و دوباره `git push`.
- با **SSH**: `git remote set-url origin git@github.com:rezabahadori76/Robot.git` سپس `git push -u origin main`.

فایل **`.gitignore`** در ریشهٔ پروژه قوانین نادیده‌گرفتن آرشیوهای لوکال (`/*.zip`, `/*.rar`)، محیط‌های مجازی، `node_modules`، فایل‌های سیستم/IDE و کپی‌های اضافهٔ لودر را تعریف می‌کند.

---

## اجرا

1. این پوشه را با یک **سرور محلی HTTP** باز کنید (به‌خاطر `fetch` برای فایل‌های OBJ/MTL و تکسچرها، باز کردن مستقیم `file://` معمولاً CORS می‌گیرد).

   ```bash
   cd Robot          # پوشهٔ ریشهٔ همین ریپو بعد از clone
   python3 -m http.server 8080
   ```

2. در مرورگر بروید به: `http://localhost:8080/` و `index.html` را باز کنید.

3. کنار `index.html` باید **`three.min.js`** (و در صورت نیاز لودرهای ارجاع‌شده در HTML) وجود داشته باشد.

---

## ساختار پروژه

| مسیر | توضیح |
|------|--------|
| `index.html` | صحنه، ناوبری، مبلمان، ویلچر، سنسور دوربین، HUD — هستهٔ برنامه |
| `three.min.js` | کتابخانهٔ Three.js (شما باید نسخهٔ سازگار را کنار فایل بگذارید) |
| `MTLLoader.js`, `OBJLoader.js`, `TDSLoader.js` | لودرهای کمکی (طبق `<script>` در HTML) |
| `assets/models/...` | مدل‌های OBJ/MTL/3DS و تکسچرها (یخچال سفارشی، مبل، گلدان، …) |

---

## قابلیت‌ها

- **چیدمان خانه:** مستطیل با دیوارهای داخلی، درها به راهرو، کف و نورپردازی.
- **مسیر:** انتخاب اتاق مبدأ/مقصد → «Show path» یا «Move wheelchair»؛ مسیر با Dijkstra روی گراف اتاق‌ها + درها، در صورت نیاز با شبکهٔ ۴همسایه دوخت می‌شود.
- **دو نما:** چپ orbit اصلی، راست **دوربین چشم‌انداز ویلچر** (`wheelCam`).
- **فاصلهٔ افقی (seat height):** پرتوهای افقی F/L/R/B از ارتفاع نشیمن برای ایدهٔ «برخورد جانبی».
- **سنسور دوربین ویلچر (object detection stub):** چند پرتو از مرکز تصویر و اطراف آن در فضای NDC دوربین ویلچر؛ برای هر آبجکت ریشه‌ای در صحنه، نزدیک‌ترین فاصله ذخیره می‌شود + **`forwardClearanceM`** برای بخش جلویی میدان دید.

---

## API جهانی: `ROBOT_HOUSE`

در کنسول مرورگر، شیء **`window.ROBOT_HOUSE`** (نسخه در `version`) شامل است:

- `rooms`, `doors`, `bounds`, `footprint`, `plan`, `nav` — دادهٔ نقشه و گراف.
- **`wheelchairCameraSensor`** — زیرساخت تشخیص مانع از دید دوربین ویلچر (بعد از لود صحنه مقداردهی می‌شود).

---

## `ROBOT_HOUSE.wheelchairCameraSensor` — تشخیص آبجکت و فاصله

این لایه **شبیه‌ساز سنسور روی دوربین** است: از **`wheelCam`** با `Raycaster.setFromCamera(ndc, wheelCam)` پرتو می‌زند، اولین برخورد معتبر با مش‌های صحنه را می‌گیرد، برخوردها را به **`root`** (نزدیک‌ترین اجداد تحت `scene`) ادغام می‌کند تا هر «آبجکت» یک فاصلهٔ مینیمم داشته باشد.

### متدها و تنظیمات

| عضو | نقش |
|-----|-----|
| `scan()` | یک اسکن کامل؛ نتیجه در `lastScan` ذخیره می‌شود. هر فریم وقتی ویلچر visible است از حلقهٔ رندر صدا زده می‌شود. |
| `lastScan` | آخرین خروجی (شکل زیر). |
| `config.maxRangeM` | حداکثر فاصلهٔ پرتو (متر). |
| `config.minAlertDistanceM` | آستانهٔ نزدیکی برای قلاب هشدار. |
| `setMaxRangeM(m)`, `setMinAlertDistanceM(m)` | تنظیم امن محدوده. |
| `setNdcSamples(samples)` | آرایهٔ جفت‌های `[[nx, ny], ...]` در فضای NDC (-1…1) برای چیدمان پرتوهای سفارشی. |
| `onNearObstacle` | setter: تابع `(lastScan) => void`. وقتی نزدیک‌ترین مانع از آستانه عبور کند **یک‌بار** (latch) فراخوانی می‌شود تا از اسپم جلوگیری شود. |
| `resetAlertLatch()` | بعد از پاک کردن مسیر، شروع حرکت جدید، یا بعد از **replan** دستی صدا بزنید تا هشدار دوباره برای نزدیک‌شدن بعدی فعال شود. |

### شکل `lastScan`

```js
{
  time: 123456.7,              // performance.now()
  detections: [               // مرتب‌شده بر حسب فاصله
    {
      rootId: "<uuid>",
      distance: 1.23,         // متر از دوربین
      point: Vector3,         // نقطهٔ برخورد جهانی
      mesh: Mesh,             // مش برخوردشده
      root: Object3D          // ریشه تحت scene
    },
    // ...
  ],
  nearest: /* همانند اولین detection یا null */,
  forwardClearanceM: 2.1,    // حداقل فاصله در بخش «جلویی» NDC یا null
  rays: [                    // یک قلم به ازای هر پرتو NDC
    { ndc: { x, y }, hit: { ... } | null }
  ],
  configSnapshot: { maxRangeM, minAlertDistanceM }
}
```

### نمونه: قلاب برای «قبل از برخورد مسیر عوض کن» (اسکلت)

```js
ROBOT_HOUSE.wheelchairCameraSensor.setMinAlertDistanceM(0.75);

ROBOT_HOUSE.wheelchairCameraSensor.onNearObstacle = function (scan) {
  const d = scan.forwardClearanceM ?? scan.nearest?.distance;
  if (d == null) return;
  console.warn("[sensor] near obstacle", d, "m", scan.nearest?.root);
  // مرحلهٔ بعدی شما:
  // - توقف انیمیشن ویلچر
  // - علامت‌گذاری سلول‌های شبکه به عنوان blocked نزدیک scan.nearest.point
  // - فراخوانی دوباره NAV.findPathDijkstra(...) از موقعیت فعلی
  // - سپس resetAlertLatch() پس از اعمال مسیر جدید
};
```

این پروژه هنوز **replan خودکار** را اجرا نمی‌کند؛ فقط **دادهٔ فاصله و آبجکت** و یک **قلاب هشدار** را فراهم می‌کند تا منطق شما را به آن وصل کنید.

### محدودیت‌ها (برای طراحی بعدی)

- تشخیص بر پایهٔ **هندسهٔ مش** است، نه کلاس semantic (مبل، دیوار، …) مگر خودتان `userData` روی مش‌ها بگذارید و در `onNearObstacle` بخوانید.
- پرتوها **رنگ/تصویر** نمی‌خوانند؛ جایگزینی با مدل یادگیری روی تصویر دوربین در آینده ممکن است.
- مدل‌هایی که دیر لود می‌شوند تا قبل از حضور در صحنه در اسکن نیستند.

---

## توسعهٔ پیشنهادی

1. **`mesh.userData`** — مثلاً `userData.obstacleClass = "furniture" | "wall"` برای فیلتر در replan.
2. **THREE.Layers** — لایهٔ جدا برای موانع قابل‌عبور vs سفت.
3. **ادغام با `NAV` / `astarGrid4`** — علامت‌گذاری سلول‌ها از روی `scan.nearest.point` در صفحهٔ xz.
4. **استخراج JS** — می‌توانید بلوک سنسور را به `wheelchair-camera-sensor.js` منتقل کنید و با `createWheelchairCameraSensor({ THREE, scene, wheelCam, ... })` تزریق وابستگی کنید.

---

## وابستگی‌ها

- مرورگر مدرن با WebGL.
- Three.js (نسخهٔ سازگار با APIهای استفاده‌شده در فایل، مثلاً `SRGBColorSpace`, `Raycaster.setFromCamera`).

---

## مجوز مدل‌ها

فایل‌های داخل `assets/models/` از منابع مختلف (مثل free3D / archibase) هستند؛ قبل از انتشار محصول، **مجوز هر مدل** را جداگانه بررسی کنید.
