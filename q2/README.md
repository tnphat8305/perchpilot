# Q2 — xác nhận bám và quyết định tắt cánh trong sim GPU (P3, phần 1) — thiết kế v2

Phần này hiện thực P3 trong `docs/PIPELINE_1_TUAN.md`, trừ phần dò lưới FSM B4 (Codex làm sau, dùng chung dataset và `sim/contact_phase.py`).

Quy ước nhãn kết quả: **RAN** = đã chạy thật, số liệu lấy từ JSON trong `results/`; **NOT RUN** = mới có lệnh hoặc job, chưa có số.

## 0. Thay đổi thiết kế v1 → v2 (2026-11-19)

**Vì sao phải đổi.** Drone đậu bằng bụng, z_thân ≈ pháp tuyến n của tấm. Vì vậy:
- lực đẩy T·z_thân **kéo đế ra khỏi tấm**, không ép vào được;
- trọng lực ép đế vào tấm (g·cosθ) và kéo dọc xuống dốc (g·sinθ).

v1 (PRESS ở 1,0·hover, sau đó PROBE hạ thrust) dựa trên ý "ép bằng thrust", nên sai về vật lý. Bằng chứng (RAN, v1): 26% episode test rơi *ngay trong lúc dò*, kể cả với oracle. Kết quả v1 vẫn còn trong lịch sử git, commit `31c49fa`.

| Mục | v1 | v2 |
|---|---|---|
| Hành động | PRESS (1,0·hover, 0,1 s) → PROBE 0,6 / 0,3 / 0,0 | **RELAX** (về idle 0,1·hover: ramp 50 ms, giữ tới 0,15 s) → **PULL 0,3 / 0,6 / 0,9** (ramp lên k·hover 50 ms, giữ 150 ms, ramp về idle 50 ms) |
| Điểm quyết định | t = 0,1 / 0,3 / 0,5 / 0,7 s | t = 0,15 / 0,40 / 0,65 / 0,90 s |
| Mất tiếp xúc khi motor còn quay (tách hoặc trượt khỏi mép) | RƠI | **SELF_ABORT**: drone đang có thrust nên bay ra, có thể đậu lại. Tính vào hủy / đậu thất bại, **không** tính là rơi. Cờ `powered_detach_is_fall=True` cho lại hành vi v1. |
| RƠI | Mọi lúc | **Chỉ sau MOTORS_OFF**: trong 10 s REST, đế tách khỏi tấm hoặc trượt ra khỏi mép |
| Load cell `f_normal_meas` | m·max(f_n, 0); khi tấm đang hút thì đọc 0 | **Có dấu**, m·(f_c·n): dương = ép, âm = kéo căng (khi PULL). Nhiễu 3% FS, với FS = 2·m·g ≈ 39 N; bão hoà ở ±FS; trễ 10 ms |
| Rung do trượt (giả định) | 10 m/s² rms cho mỗi m/s (gyro 0,5) | **2 m/s² rms cho mỗi m/s** (gyro 0,1) |
| Nhãn chính `held_star` | margin ≥ 0 tại mọi bước trong 10 s | Đế còn trên tấm suốt 10 s **và tổng quãng trượt < 2 cm**. Nhãn v1 giữ lại dưới tên `held_strict` |
| False-safe chính | MOTORS_OFF và held*=0 / số MOTORS_OFF | **Rơi sau MOTORS_OFF / số MOTORS_OFF**. Báo kèm bản theo `held_star` và theo `held_strict` |
| Chọn τ* | false-safe ≤ 2% trên val | false-safe ≤ **1%** trên val: chừa biên an toàn dưới mục tiêu 2%, vì v1 cho thấy ngưỡng không chuyển tốt sang vật liệu chưa thấy |
| B7 oracle | MOTORS_OFF ở điểm quyết định đầu tiên có held*=1 | Biết μ, F_hold nên **MOTORS_OFF ngay lúc chạm** nếu held_star = 1, ngược lại ABORT. Bản "oracle trên lịch dò" vẫn được báo |
| Baseline mới | — | "Tắt motor ngay khi chạm, không kiểm tra" |

## 1. Các tệp

| Tệp | Nội dung |
|---|---|
| `sim/contact_phase.py` | Simulator pha sau khi chạm, 200 Hz, vector hoá theo N môi trường, chạy CPU hoặc CUDA. Có macro RELAX / PULL(k) / MOTORS_OFF / ABORT, 11 kênh cảm biến (C5), nhãn tính bằng rẽ nhánh 10 s tắt motor (`rest_fork`). |
| `q2/test_contact_phase.py` | Tự kiểm tra (CPU, khoảng 3 s): margin trùng `hold_margin`; thrust dọc n làm giảm f_n đúng 1:1; rẽ nhánh dạng đóng trùng 100% với bước mô phỏng thật (fell, held_strict, held_star); load cell ra dấu + khi RELAX, dấu − khi PULL 0,9; không bao giờ RƠI khi motor còn quay. |
| `q2/dataset.py` | Sinh cửa sổ 80 × 11 (float16, chia shard) tại các điểm quyết định, chia train/val (vật liệu đã thấy) và test (vật liệu chưa thấy). |
| `q2/belief_net.py` | 1D-CNN (3 lớp conv, 64 kênh), ba đầu ra p_hold / p_slip / margin; các hàm AUROC, ECE, Brier, Wilson. |
| `q2/train_belief.py` | Huấn luyện, temperature scaling trên val, chấm trên test, xuất TorchScript và JSON kết quả. Chọn nhãn bằng `--label held` (mặc định, held_star), `strict` hoặc `nofall`. |
| `q2/decide.py` | (v2) Luật D-thr, chọn τ* trên val, chấm vòng kín trên episode test, đường cong rủi ro–hiệu quả, các baseline, kiểm tra vòng kín online. |
| `q2/decide_v3.py` | (v3, mục 10) Lịch dò thích nghi (dừng sớm theo p_slip), ensemble và luật "phải qua PULL 0,3", FSM B4a/B4b với dò lưới, bảng H2 ở cùng false-safe, kiểm tra vòng kín online. |

Dữ liệu và checkpoint **không** nằm trong repo:
- dataset: `/home/sup/data/perchpilot/q2data/<tag>/`;
- checkpoint: `/home/sup/data/perchpilot/runs/q2_belief_<tag>/` (`best.pt`, `belief.ts`, `belief.json`).

Chỉ commit JSON nhỏ trong `results/`: `q2_belief_<tag>.json`, `q2_decide_<tag>.json`, `q2_decide_<tag>_curve.json`.

## 2. Cách chạy

```bash
cd perchpilot
PY=/home/sup/data/venvs/occwm/bin/python
$PY q2/test_contact_phase.py
$PY q2/dataset.py --out /home/sup/data/perchpilot/q2data/v2_small --n_train 60000 --n_val 10000 --n_test 10000 --chunk 1024
$PY q2/train_belief.py --data /home/sup/data/perchpilot/q2data/v2_small --tag v2_small --epochs 12
$PY q2/decide.py --belief /home/sup/data/perchpilot/runs/q2_belief_v2_small --tag v2_small --n_val 8192 --n_test 8192
```

Tuỳ chọn hay dùng:
- `--drop imu|fnormal|contact|imu,fnormal`: bỏ nhóm kênh, dùng cho ablation.
- `--cfg key=val,...` (trong `dataset.py`): đổi tham số `ContactCfg`. `decide.py` tự đọc lại cấu hình này từ `meta.json` của dataset.
- `--device cuda`: chạy trên GPU.

Bản đầy đủ (200k / 20k / 20k) chạy qua hàng đợi GPU, xem mục 9.

## 3. Mô hình vật lý pha tiếp xúc (giả định)

Hệ quy chiếu là hệ hướng của xe (z lên). `hold_margin` bất biến khi quay quanh z, nên kết quả tính trong hệ này bằng kết quả tính trong hệ thế giới của `quad_perch_env`.

**Điều kiện ban đầu lúc chạm.** Hiện được lấy mẫu tham số trong đường bao chạm đạt:

| Đại lượng | Miền |
|---|---|
| v_n | 0,2–1,5 m/s |
| v_t | 0–1 m/s, hướng ngẫu nhiên trong mặt tấm |
| Lệch trục z_thân so với n | 0–25°, phương vị đều |
| Body rate | ≤ 5 rad/s |
| Góc tấm | 55–70° |
| Tốc độ xe | 0–3 m/s |
| Thrust lúc chạm | 0,6–1,2 T_hover |
| Điểm chạm | đều trong ±0,15 m quanh tâm tấm |

**Sau này các trạng thái chạm này sẽ được thay bằng trạng thái chạm lấy từ rollout của chính sách Q1 đã huấn luyện.** Khi đó dùng đúng các trường trên.

**Gia tốc xe.** Chép nguyên `QuadPerchEnv._step_vehicle`, chạy với dt = 0,005 s. Chuỗi gia tốc được sinh trước cho toàn bộ khoảng: lịch sử 0,4 s + tối đa 1,2 s tiếp xúc + 10 s REST. Nhờ vậy nhãn và kết quả vòng kín dùng **cùng** tương lai của xe.

**Lực.** Mọi đại lượng tính trên một đơn vị khối lượng. Quyết định giữ / trượt / tách luôn lấy từ `hold_margin`, import trực tiếp.
- `extra_acc = a_xe + a_va_chạm`, với `a_va_chạm(t) = (v_n·n − v_t·t̂)/τ_imp · exp(−t/τ_imp)`, τ_imp = 20 ms (**giả định**).
- `thrust_acc = T·z_thân`, T bám theo lệnh qua trễ bậc nhất τ_T ∈ [0,02; 0,05] s.
- `normal_ok = hold_margin(mu=10⁶, …)`: với μ rất lớn hàm trả về đúng `normal_ok`. `normal_ok < 0` → tách. `margin < 0` → trượt theo −f_t với tốc độ `slip_gain·|margin|·m` (slip_gain = 0,05 (m/s)/N theo C2, m = 2 kg, **giả định**). Đế ra khỏi tấm (nửa cạnh 0,22 m tính từ tâm) → mất tiếp xúc.
- Mất tiếp xúc khi motor còn quay → SELF_ABORT. Sau MOTORS_OFF → RƠI.
- Đế cứng: tư thế drone so với tấm không đổi khi đang bám. Trượt không làm đổi margin, nên 10 s REST tính được ở dạng đóng, vector hoá theo cả trục thời gian. Bộ tự kiểm tra xác nhận dạng đóng trùng 100% với bước mô phỏng thật.

**Ý nghĩa vật lý của PULL(k)** (mặc định θ ≈ 62°, lệch nhỏ):
- f_n ≈ g·cosθ − k·g ≈ (0,47 − k)·g. PULL 0,6 và 0,9 làm f_n âm, tức kéo căng, và thử trực tiếp F_hold (cần F_hold > k − 0,47 theo đơn vị g).
- Lực ma sát tối đa giảm từ μ·(g·cosθ + F_hold) xuống μ·F_hold, trong khi tải tiếp tuyến g·sinθ không đổi. Vì vậy PULL thử μ·F_hold.
- Đế chịu được PULL(k) nghĩa là khi tắt motor còn dư khoảng k·g lực pháp tuyến cho các gia tốc xe sau này.

## 4. Cảm biến (C5, 200 Hz, 11 kênh)

| Kênh | Mô hình |
|---|---|
| `acc_body` (3) | Lực riêng `a − g` trong hệ thân, cộng gia tốc trượt. Nhiễu trắng 0,007 m/s² (ICM-42688: 70 µg/√Hz × √100 Hz). Rung motor 0,4 m/s² rms ở thrust hover, tỉ lệ với thrust. Rung do trượt **2 m/s² rms cho mỗi m/s tốc độ trượt** (**giả định chưa hiệu chỉnh**). Bias mỗi episode σ = 0,05. |
| `gyro_body` (3) | Tốc độ yaw của xe đưa về hệ thân, cộng body rate lúc chạm tắt dần với τ = 30 ms. Nhiễu 0,0005 rad/s. Rung motor 0,02 rad/s. Rung do trượt 0,1 rad/s cho mỗi m/s. Bias σ = 0,005. |
| `f_normal_meas` (1, N) | **Có dấu**: m·(f_c·n), dương = ép, âm = kéo căng. Trễ 10 ms, nhiễu σ = 3% FS (FS = 2·m·g ≈ 39 N, σ ≈ 1,2 N), bão hoà ±FS. Bằng 0 (chỉ còn nhiễu) sau khi tách. |
| `contact` (1) | 1 khi đang gắn. Mỗi mẫu có 0,5% khả năng rớt về 0. |
| `thrust_cmd01` (1) | T_cmd/(TW·g), theo quy ước C1. |
| `phase_id` (1) | 0 tiếp cận, 1 RELAX, 2 PULL, 3 OFF, 4 ABORT. |
| `probe_level` (1) | 0,1 khi RELAX; k khi PULL(k); 1,0 khi tiếp cận. |

0,4 s trước khi chạm là pha bay tự do: lực riêng (0, 0, T), gyro bằng body rate lúc chạm, contact = 0.

## 5. Nhãn (chỉ dùng để huấn luyện và chấm, không bao giờ là đầu vào)

Mọi nhãn tính bằng cách tắt motor ngay tại bước hiện tại (thrust tắt dần theo τ_T) rồi mô phỏng 10 s tiếp theo với gia tốc tương lai thật của xe.

- `held_star` (nhãn chính v2): không tách, đế còn trên tấm, tổng quãng trượt < 2 cm.
- `held_strict` (nhãn v1, đúng chữ đặc tả §5.1): `hold_margin ≥ 0` tại mọi bước.
- `fell_if_off`: tách ra hoặc trượt khỏi tấm trong 10 s.
- Kèm theo: `slip_path_if_off_m`, `margin_min_g`, `slip_now` (margin < 0 tại bước hiện tại).

## 6. Cảm biến thực sự mang thông tin gì

**Load cell đo lực tiếp xúc ròng, không đo trực tiếp F_hold hay μ.** Giá trị m·(f_c·n) chỉ do thrust, hình học và gia tốc xe quyết định. Điểm khác so với v1: nhờ có dấu và nhờ PULL, load cell cho biết *tải kéo căng* mà đế đang chịu. Kết hợp với sự kiện còn bám / trượt / tách, ta có cận về F_hold và μ·F_hold.

**IMU khi đang bám cứng** đo `a_tấm − g`, cũng không phụ thuộc μ hay F_hold.

**Thông tin về vật liệu vẫn chỉ lộ ra qua sự kiện:** trượt hoặc không ở RELAX và ở từng mức PULL; trượt ngay lúc va chạm; tách ra. Trượt được "nhìn thấy" qua rung do trượt trên IMU (giả định 2 m/s² cho mỗi m/s). Tách được nhìn thấy qua contact và load cell về 0.

Câu hỏi "bỏ IMU thì còn dự báo được không" được kiểm tra bằng ablation ở mục 7.

## 7. Kết quả quy mô nhỏ v2 (RAN, CPU)

**Cấu hình chạy:**
- dataset `v2_small`: 60k / 10k / 10k cửa sổ, mỗi episode 4 điểm cố định + 2 điểm ngẫu nhiên;
- mạng niềm tin: 12 epoch trên CPU (ablation 10 epoch);
- `decide.py`: 8192 episode val + 8192 episode test (ablation 4096 + 4096);
- peak RSS 0,7–1,1 GB, trong đó riêng `import torch` đã khoảng 0,55 GB.

Mọi tỉ lệ có kèm Wilson 95%. Nguồn số liệu: `results/q2_belief_v2_small*.json`, `results/q2_decide_v2_small*.json` (+ `_curve.json`).

### 7.1 Cân bằng lớp (theo cửa sổ) và mất tiếp xúc khi đang dò

| Tập | Số cửa sổ | held_star | held_strict | fell_if_off | slip_now | SELF_ABORT trong lịch dò (theo episode) |
|---|---|---|---|---|---|---|
| train (vật liệu đã thấy) | 60 000 | 0,786 [0,782; 0,789] | 0,702 | 0,179 | 0,288 | 20,0% (đều do trượt khỏi mép, 0% do tách) |
| val (đã thấy) | 10 000 | 0,783 [0,775; 0,791] | 0,696 | 0,186 | 0,295 | 23,3% |
| test (chưa thấy) | 10 000 | 0,706 [0,697; 0,715] | 0,618 | 0,246 | 0,362 | 29,0% (1,5% do tách, 27,5% do trượt khỏi mép) |

held_star tại 4 điểm quyết định (train, chỉ tính episode còn gắn): 0,72 / 0,76 / 0,83 / 0,88. Tỉ lệ tăng dần một phần vì các ca yếu đã trượt khỏi tấm trong lúc PULL.

### 7.2 Mạng niềm tin trên test (vật liệu chưa thấy)

| Mô hình | AUROC hold | ECE-15 | Brier | AUROC slip |
|---|---|---|---|---|
| Đủ 11 kênh, nhãn held_star | **0,975** | 0,020 | 0,057 | 0,996 |
| Đủ kênh, nhãn nofall | 0,979 | 0,021 | 0,049 | 0,996 |
| Đủ kênh, nhãn held_strict (v1) | 0,987 | 0,016 | 0,037 | 0,995 |
| Bỏ f_normal | 0,971 | 0,019 | 0,062 | 0,995 |
| Bỏ contact | 0,972 | 0,023 | 0,061 | 0,994 |
| **Bỏ IMU** | **0,607** | 0,077 | 0,208 | 0,576 |
| **Bỏ IMU + f_normal** | **0,593** | 0,076 | 0,209 | 0,532 |

Trên val, mô hình đủ kênh đạt AUROC 0,986, ECE 0,008, T = 1,07.

**Đọc kết quả cho trung thực:**
- **Kỳ vọng "bỏ IMU thì AUROC không sụp nhờ load cell khi PULL" KHÔNG đúng.** Bỏ IMU: AUROC còn 0,61. Bỏ cả IMU lẫn load cell: 0,59. Load cell chỉ thêm khoảng 0,01.
- Lý do: load cell đo lực tiếp xúc ròng m·(f_c·n). Đại lượng này do thrust, hình học và gia tốc xe quyết định, **dù đế đang bám hay đang trượt**. Nó chỉ đổi khi đế *tách*, nhưng với vật liệu đã thấy (F_hold ≥ 0,6 mg > 0,9 − 0,47) thì PULL 0,9 gần như không làm tách (0% trên train). Phản ứng của vật liệu khi PULL là *trượt*, mà trượt chỉ thấy được qua IMU trong mô hình này.
- Muốn load cell có ích thì cần cảm biến đo **lực cắt / tiếp tuyến** (load cell 3 trục: tỉ số f_t/f_n khi bắt đầu trượt cho thông tin về μ), hoặc cảm biến dịch chuyển của đế (optical flow dưới đế, encoder). Cả hai chưa có trong C5.
- "Lối tắt": riêng bit thật "đang không trượt" đã cho AUROC 0,91–0,93 với held_star (v1 là 0,95–0,96). Mạng vẫn chủ yếu là bộ phát hiện trượt, nhưng nhãn mới bớt bị chi phối bởi vi trượt.

### 7.3 Bộ quyết định, chạy vòng kín trên episode

τ* chọn trên val với false-safe (rơi) ≤ 1%, rồi áp nguyên lên test (vật liệu chưa thấy).

| Luật (test, 8192 episode) | Đậu hợp lệ | False-safe (rơi) | FS (held_star) | FS (strict) | Hủy | SELF_ABORT |
|---|---|---|---|---|---|---|
| **D-thr**, τ* = 0,81 | **0,595** [0,584; 0,606] | **0,021** [0,017; 0,025] | 0,055 | 0,096 | 0,117 | 0,253 |
| D-thr, mô hình nhãn nofall, τ* = 0,93 | 0,602 [0,592; 0,613] | 0,022 [0,018; 0,026] | 0,063 | 0,110 | 0,109 | 0,248 |
| D-thr, τ chọn ở 2% không biên (0,54) | 0,611 | 0,048 | 0,097 | 0,152 | 0,086 | 0,238 |
| D-thr, τ chọn theo held_star ≤ 1% (0,97) | 0,546 | **0,004** | 0,021 | 0,044 | 0,175 | 0,267 |
| D-thr bỏ PULL (quyết định sau RELAX, không thì ABORT), τ = 0,61 | 0,585 [0,575; 0,596] | 0,020 | 0,051 | 0,080 | 0,371 | 0,013 |
| **B7 oracle** (MOTORS_OFF ngay lúc chạm nếu held_star) | **0,616** [0,605; 0,626] | 0 | 0 | 0,084 | 0,384 | 0 |
| B7 oracle trên lịch dò | 0,632 | 0 | 0 | 0,083 | 0,129 | 0,239 |
| B7-static (không biết gia tốc xe tương lai) | 0,629 | 0,028 | 0,088 | 0,159 | 0,096 | 0,214 |
| **Tắt motor ngay khi chạm, không kiểm tra** | 0,616 | **0,311** | 0,384 | 0,436 | 0 | 0 |
| Luôn tắt motor sau RELAX | 0,630 | 0,305 | 0,362 | 0,413 | 0 | 0,013 |

Trên val:
- D-thr: đậu hợp lệ 0,697, false-safe 0,010;
- bỏ PULL: 0,685 / 0,010;
- B7 lúc chạm: 0,675;
- tắt ngay không kiểm tra: false-safe 0,240.

**Kiểm tra vòng kín online.** Chạy lại bằng `ContactPhaseSim.act/step`, mạng niềm tin trong vòng lặp, 1024 episode test: trùng **100%** với bản đánh giá offline, cho mọi mô hình.

**Ablation kênh ở tầng quyết định:**
- Bỏ f_normal hoặc bỏ contact: đậu hợp lệ trên test gần như không đổi (0,59), nhưng false-safe trên test lên 3,0–3,1%.
- Bỏ IMU: không tìm được τ hữu ích, đậu hợp lệ ≈ 0.

**Kết luận v2:**
1. **Bước xác nhận là cần thiết.** Tắt motor ngay không kiểm tra cho 31% rơi; D-thr giảm xuống 2,1% mà vẫn giữ 97% số lượt đậu hợp lệ của oracle (0,595 so với 0,616).
2. **False-safe trên vật liệu chưa thấy vẫn vượt 2%** (2,1% [1,7; 2,5]), dù τ* được chọn ở 1% trên val. Theo nhóm trên test: μ thấp (0,15–0,25, ngoài miền huấn luyện) cho false-safe **29%** trên 310 lượt tắt motor; μ cao chỉ 0,4%. Mô hình **tự tin quá mức với vật liệu trơn hơn mọi vật liệu đã thấy**. Cách xử lý (chưa làm): mở rộng miền μ khi huấn luyện, chọn τ theo held_star (false-safe theo rơi 0,4%, đổi lại đậu hợp lệ còn 0,546), hoặc ước lượng bất định / phát hiện ngoài phân phối.
3. **PULL chỉ giúp rất ít:** đậu hợp lệ +1 điểm (0,595 so với 0,585, khoảng tin cậy chồng nhau). Cái giá là 25% lượt SELF_ABORT do đế trượt khỏi mép khi PULL 0,9, so với 1,3% khi chỉ RELAX. Phần lớn các lượt này vốn cũng không bám được, nên đậu hợp lệ không giảm. Nhưng mỗi SELF_ABORT là một lần phải bay lại. Nên thử: dừng PULL ngay khi phát hiện trượt, hoặc bỏ PULL 0,9.

## 8. Rủi ro mở

1. **Mô hình rung do trượt chưa hiệu chỉnh** (`acc_vib_slip` = 2, `gyro_vib_slip` = 0,1). Đây vẫn là nguồn thông tin chính. Cần đo trong Gazebo (plugin HoldPad) hoặc trên phần cứng.
2. **Load cell 1 trục không mang thông tin vật liệu** trong mô hình này (mục 7.2). Cân nhắc đổi C5 sang load cell 3 trục hoặc thêm cảm biến dịch chuyển đế.
3. **Vật liệu trơn ngoài miền huấn luyện** làm false-safe vượt 2% (mục 7.3).
4. **Trạng thái chạm đang là mẫu tham số**, chưa lấy từ rollout Q1.
5. **Các giả định mô hình:** đế cứng, không bập bênh, không đàn hồi; va chạm là xung mũ τ = 20 ms; m = 2 kg; FS load cell = 2·m·g.
6. **FSM B4 (Codex) chưa có**, nên chưa so sánh được H2. Codex cần dùng lịch dò v2 và các điểm quyết định t = 0,15 / 0,40 / 0,65 / 0,90 s.

## 9. Bản đầy đủ — NOT RUN (đã xếp hàng GPU, chạy sau P2)

Các job chạy với thư mục làm việc là thư mục gốc của mã nguồn.

| Job | Việc |
|---|---|
| `40_q2_data.job` | Sinh dataset 200k / 20k / 20k vào `q2data/v2_full`, CUDA. Vẫn dùng lịch dò v2, vì lịch v2 bao trùm lịch v3. |
| `41_q2_belief.job` | (v3) Huấn luyện 5 mạng nhãn held_star, seed 0–4, để làm ensemble; 25 epoch mỗi mạng. |
| `42_q2_belief_ablation.job` | Ablation: bỏ fnormal / imu / contact / imu+fnormal (seed 0). |
| `43_q2_decide.job` | (v3) `decide_v3.py` với ensemble 5 mạng: mọi biến thể D-thr, FSM B4a/B4b, bảng H2, 20k + 20k episode. Sau đó chạy `decide_v3.py` một mạng cho từng mô hình ablation. |

## 10. Thiết kế v3 — lịch dò thích nghi, bảo thủ ngoài miền, FSM B4 (RAN, CPU)

Mã: `q2/decide_v3.py`. Kết quả: `results/q2_decide_v3_v2_small_ens5.json` (+ `_curve.json`). Các mạng ensemble: `results/q2_belief_v2_small{,_s1,_s2,_s3,_s4}.json`.

Vật lý, cảm biến và dataset giữ nguyên như v2. Dataset v2 sinh theo lịch v2, mà lịch v3 là đoạn đầu của lịch v2, nên dùng lại được. Mọi biến thể được chấm trên **cùng** các episode (8192 val + 8192 test): cùng vật lý, cùng nhiễu, cùng tương lai của xe.

**Luật v3.** Lịch dò RELAX → PULL 0,3 → PULL 0,6 (bỏ PULL 0,9). Tại mỗi điểm quyết định, theo thứ tự:
1. Nếu p_slip ≥ s*: dừng kéo, ABORT có kiểm soát. Trong mô phỏng tính là ABORT ngay tại điểm quyết định.
2. Ngược lại, nếu p_hold ≥ τ*: MOTORS_OFF.
3. Ngược lại: sang macro tiếp theo; hết lịch thì ABORT.

**Cách chọn (s\*, τ\*) và ngưỡng FSM trên val:**
- chỉ xét các cấu hình có false-safe (rơi) ≤ 1%;
- trong đó, các cấu hình có đậu hợp lệ cách giá trị tốt nhất không quá 1 điểm được coi là ngang nhau;
- trong nhóm ngang nhau, chọn cấu hình ít SELF_ABORT nhất, rồi đến ngưỡng bảo thủ nhất.

**Vì sao dừng ở điểm quyết định là đủ (RAN, 2048 episode mỗi tập).** Gần như mọi lần mất tiếp xúc trong một macro đều rơi vào episode **đã đang trượt** ở điểm quyết định ngay trước đó: val 106/107, 177/177, 143/146; test 171/172, 191/193, 195/198. Vì vậy không cần giám sát liên tục trong lúc PULL.

### 10.1 Ba lịch dò trên test

8192 episode vật liệu chưa thấy, một mạng (seed 0).

| Biến thể | (τ*, s*) | Đậu hợp lệ | False-safe (rơi) | FS (held_star) | Hủy | SELF_ABORT |
|---|---|---|---|---|---|---|
| Lịch v2 cũ (đến PULL 0,9) | 0,82; — | 0,594 [0,583; 0,604] | 0,020 [0,017; 0,025] | 0,054 | 0,118 | **0,254** |
| Lịch v3, không dừng sớm | 0,76; — | 0,598 [0,587; 0,609] | 0,019 [0,016; 0,023] | 0,053 | 0,185 | 0,183 |
| **Lịch v3, có dừng sớm** | 0,56; 0,6 | 0,586 [0,576; 0,597] | 0,021 [0,018; 0,025] | 0,051 | 0,369 | **0,013** |

Trên val, ba biến thể lần lượt đạt (đậu hợp lệ / false-safe / SELF_ABORT): 0,696 / 0,009 / 0,191; 0,697 / 0,010 / 0,126; 0,687 / 0,010 / 0,005.

**Kết luận:** dừng sớm khi phát hiện trượt cắt SELF_ABORT từ 25% xuống 1,3%. False-safe không đổi. Cái giá là khoảng 1 điểm đậu hợp lệ, vì một ít ca bị dừng nhầm. Các ca bị dừng trở thành ABORT có kiểm soát, thay cho việc trượt khỏi mép.

### 10.2 Bảo thủ ngoài miền (μ trơn hơn miền huấn luyện)

Trên test, nhóm μ 0,15–0,25 chiếm khoảng 1/3 số episode. Miền μ huấn luyện **không** được mở rộng.

| Biến thể (test) | Đậu hợp lệ toàn tập | False-safe toàn tập | False-safe nhóm μ 0,15–0,25 (số lượt tắt motor) | False-safe nhóm μ 1,0–1,2 |
|---|---|---|---|---|
| v3 dừng sớm, 1 mạng | 0,586 | 0,021 [0,018; 0,025] | **0,327** [0,274; 0,385] (269) | 0,004 |
| (a) + ensemble 5 mạng, p_hold = min | 0,590 | 0,022 [0,019; 0,027] | 0,320 [0,270; 0,375] (300) | 0,004 |
| (b) + phải qua PULL 0,3 (val chọn s = tắt) | 0,608 | 0,028 [0,024; 0,033] | 0,296 [0,251; 0,345] (365) | 0,008 |
| (a) + (b) | 0,603 | 0,025 [0,021; 0,030] | 0,283 [0,238; 0,332] (350) | 0,007 |
| B7 oracle lúc chạm | 0,616 | 0 | 0 (106) | 0 |

**Kết luận (âm tính, báo thật):** cả (a) lẫn (b) **không** làm giảm rõ overconfidence trên vật liệu trơn chưa thấy. False-safe nhóm μ thấp vẫn 28–33%.
- Lý do (a) không giúp: các thành viên ensemble học cùng một dữ liệu, nên cùng sai theo một cách. Min p_hold gần bằng p_hold của từng mạng.
- Lý do (b) không giúp: khi bắt buộc qua PULL 0,3, việc chọn trên val bù lại bằng τ thấp hơn (0,54 so với 0,56; val chọn tắt luôn dừng sớm), nên số lượt tắt motor ở μ thấp còn tăng.
- Gốc vấn đề:
  - "không trượt lúc RELAX / PULL nhỏ" chỉ là cận dưới của margin;
  - với μ trơn hơn mọi vật liệu đã thấy, margin thường mỏng, và gia tốc xe về sau đủ để làm trượt;
  - ngay cả oracle biết μ và F_hold nhưng không biết gia tốc tương lai (B7-static, v2) cũng có false-safe 2,8%.
- Hướng còn lại, chưa làm:
  - chọn τ ở mức chặt hơn trên val. Theo đường cong test, D-thr ở false-safe ≤ 0,5% vẫn còn 0,55 đậu hợp lệ;
  - phát hiện ngoài phân phối từ f_normal / hình học;
  - PULL mạnh hơn nhưng có dừng sớm.

### 10.3 FSM đối chứng B4, dùng cùng cảm biến và cùng lịch dò

Ngưỡng dò lưới trên val, chọn theo cùng luật với D-thr (false-safe ≤ 1%).
- **B4a** (không dò, quyết định khi kết thúc RELAX): contact bật liên tục ≥ t_c, |f_normal| ≥ F_min, ‖gyro‖ < g_max trong 0,1 s. Lưới 4 × 6 × 7. Chọn được t_c = 0,025 s, F_min = 0, g_max = 0,02 rad/s.
- **B4b** (lịch v3): phát hiện trượt bằng năng lượng của sai phân bậc một (lọc thông cao) của acc_x, acc_y trong macro vừa xong; vượt e_thr thì ABORT. Không trượt ở bước nào thì MOTORS_OFF sau PULL 0,6. Chọn được e_thr = 0,66 (m/s²)².
- Đo trên test, chất lượng phát hiện trượt: AUROC của năng lượng HP là 0,968; AUROC của p_slip từ mạng niềm tin là 0,996.

| Luật (test) | Đậu hợp lệ | False-safe (rơi) | Hủy | SELF_ABORT |
|---|---|---|---|---|
| D-thr v3 dừng sớm | **0,586** [0,576; 0,597] | 0,021 [0,018; 0,025] | 0,369 | 0,013 |
| B4a (không dò) | 0,000 [0,000; 0,001] | 0/3 lượt OFF (Wilson 95% [0%; 56,15%]) | 0,987 | 0,013 |
| B4b (có dò) | 0,496 [0,486; 0,507] | 0,026 [0,022; 0,032] | 0,463 | 0,019 |
| Tắt motor ngay khi chạm, không kiểm tra | 0,616 | 0,311 | 0 | 0 |
| B7 oracle lúc chạm | 0,616 | 0 | 0,384 | 0 |

**Bảng H2 mô tả v3** (`h2_same_false_safe_test`). Bảng lấy tỷ lệ đậu hợp lệ tốt nhất trên đường cong **test** ở cùng mức false-safe. τ của D-thr được quét với s* cố định như đã chọn. Xem giao thức kiểm định v4 ở mục 11.

| False-safe ≤ | D-thr v3 dừng sớm | D-thr v3, ensemble | B4b | B4a | D-thr − B4b |
|---|---|---|---|---|---|
| 0,5% | 0,558 | 0,553 | 0,482 | 0 | **+7,6 điểm** |
| 1% | 0,574 | 0,573 | 0,490 | 0 | **+8,4 điểm** |
| 2% | 0,586 | 0,589 | 0,494 | 0 | **+9,2 điểm** |

Nếu chỉ dùng ngưỡng chọn trên val, D-thr hơn B4b 9,0 điểm, và false-safe của D-thr còn thấp hơn (2,1% so với 2,6%).

**H2 theo mức đánh giá trong `PIPELINE_1_TUAN.md`:**
- Trên đường cong test mô tả, D-thr hơn B4b khoảng 8–9 điểm ở cùng false-safe.
- B4a không vận hành được ở false-safe ≤ 2%. Contact, lực pháp tuyến và gyro khi RELAX không chứa thông tin về độ bám. Đây là đúng lập luận của đề tài: FSM không dò thì không xác nhận được bám.
- Toàn bộ kết quả vẫn phụ thuộc giả định rung do trượt (mục 8.1). B4b cũng dựa trên tín hiệu này.

**Kiểm tra vòng kín online** cho chính sách v3 dừng sớm + ensemble, 1024 episode test: trùng **100%** với bản đánh giá offline.

## 11. Q2 v4 — giao thức kiểm định H2

V4 dùng đặc tả v1.1. False-safe chính là rơi sau MOTORS_OFF / số lượt OFF. Báo kèm `false_safe_held`, `false_safe_strict` và tỷ lệ OFF. Tỷ lệ false-safe trên đường cong là JSON `null` khi có 0 lượt OFF. [Đặc tả §5 và §8](../../docs/DAC_TA_PerchPilot.md#5-q2--xác-nhận-bám-và-quyết-định-tắt-cánh)

1. Sinh dữ liệu bằng `q2/dataset.py --protocol v4`. Kích thước tính bằng **episode**. Train và val trong miền được lưu dưới dạng shard. Dev_ood và test cùng phân phối ngoài miền, có các dải seed/ID riêng trong `meta.json`. Trình chấm sinh episode của từng tập theo kế hoạch này khi đến lượt sử dụng.
2. Chạy `q2/train_belief.py` với dataset v4 để huấn luyện trên train và chọn checkpoint, hiệu chỉnh trên val. Dùng `--drop` cho ablation. Kênh `thrust_cmd01` là lệnh chuẩn hóa đã gửi dựa trên hover danh nghĩa TW 2,26.
3. Chạy `q2/decide_v4.py --data <dataset> --belief <run_s0> <run_s1> ... --tag <tag>`. Có thể thêm `--ablations <run_drop1> ...` trong cùng lượt. D-thr chính dùng seed nhỏ nhất. Từng seed có bộ ngưỡng riêng. B4b giữ luật v3, B4c có ngưỡng OFF theo pha và B4d dùng logistic regression trên đặc trưng cửa sổ 80×11. D-thr-late được quyền OFF sau PULL 0,6, cùng thời điểm với B4b.
4. Với từng trần 0,5%, 1%, 2%, chọn cấu hình tối đa hóa đậu hợp lệ trong các cấu hình có cận trên Wilson 95% trên dev_ood đạt trần. Nếu hòa, ưu tiên cận trên thấp hơn, rồi ít SELF_ABORT hơn. Lưu `results/q2_v4_<tag>_lock.json` và hash SHA-256 trước khi sinh test. Lượt test dùng chung episode và đánh dấu `exceeds_cap` theo tỷ lệ quan sát được. Lượt kiểm tra online chạy trên batch dev_ood đầu tiên.
5. Đọc `results/q2_v4_<tag>.json` để lấy CI Wilson, CI bootstrap ghép cặp theo episode (10.000 mẫu) và biến thiên seed. `null` tại một ô test cho biết dev_ood chưa có cấu hình đủ điều kiện. `episode_archive` trỏ tới NPZ chứa ID, nhóm vật liệu và cờ kết quả theo (trần, luật, episode). Trường `available` cho biết ô có cấu hình khóa. Trường `off_time_s` là NaN với episode không OFF.

`q2/sensitivity_q2.py` chạy sáu tổ hợp `vib_scale` {1, 0,5, 0} × `slip_acc_in_imu` {1, 0}. Mỗi tổ hợp huấn luyện lại mạng và B4d, dò lại mọi ngưỡng trên dev_ood rồi chấm test đã khóa. `vib_scale` tác động lên rung do trượt. Gia tốc trượt đo được bởi IMU là tín hiệu có thật về vật lý. Biên độ và độ trễ của tín hiệu này chưa được hiệu chuẩn. Kết quả nằm ở `results/q2_v4_sens_*.json`.

Để tái lập v2/v3, dùng checkpoint tương ứng với `decide.py` hoặc `decide_v3.py`. `dataset.py --protocol legacy` sinh shard theo quy ước cũ.

Lệnh CPU, dòng job GPU C4 và kết quả chạy nhỏ nằm trong [báo cáo Q2 v4](../../research/codex/PIPE_D3_q2_v4.md).
