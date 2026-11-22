# PerchPilot — mã nguồn và cách tái lập

Đề tài: drone học **đậu bám agile** lên tấm nghiêng 55–70° đặt trên xe đang chạy (Q1), **xác nhận bám trước khi tắt motor** (Q2), và chạy trọn **chu kỳ đậu–nghỉ–rời** (Q3), hoàn toàn trong mô phỏng (sim GPU + Gazebo Harmonic + PX4 SITL).

Nhãn trạng thái dùng trong README: **ĐÃ CHẠY**, **CHƯA XONG**.

## 1. Môi trường

| Thành phần | Phiên bản / vị trí |
|---|---|
| GPU | RTX 4060 Laptop 8 GB (toàn bộ số liệu đo trên máy này) |
| Python huấn luyện | `/home/sup/data/venvs/occwm/bin/python` (CPython 3.12.3, torch 2.15 + CUDA, numpy 1.26.4). Gọi tắt là `$PY` bên dưới |
| ROS 2 | Jazzy (`/opt/ros/jazzy`), workspace px4_msgs `~/ws_px4` |
| Gazebo | Harmonic 8.11 |
| PX4 | v1.15.4. Bản vá perch mode build trong một bản sao riêng, `/home/sup/data/perchpilot/PX4-perch` (`px4_patches/0001-perch-mode-land-detector.patch`) |
| Dữ liệu lớn | `/home/sup/data/perchpilot/` gồm `runs/`, `queue/`, `sitl_eval/`, `q2data/`. Repo không chứa dữ liệu, checkpoint hay kết quả |

Mọi lệnh dưới đây chạy ở thư mục gốc của repo.

## 2. Phần tự làm và phần dùng sẵn

| Tự làm (trong repo này) | Dùng sẵn |
|---|---|
| Simulator GPU `sim/quad_perch_env.py` và `sim/quad_perch_env_v2.py` (camera FOV, nhiễu odometry, Kalman, ρ, thưởng nhận thức theo công thức của PerchRL) | PyTorch |
| PPO `rl/ppo.py` (critic bất đối xứng, khởi động ấm mở rộng chiều, kẹp std) | — |
| Pha tiếp xúc Q2 `sim/contact_phase.py`, mạng niềm tin và luật quyết định `q2/` | — |
| Plugin Gazebo `gz/plugins/hold_pad/` (C++), model `gz/models/x500_perch`, world `gz/worlds/perch_world.sdf` | Gazebo Harmonic, model x500 của PX4 |
| Bản vá PX4 perch mode (MAV_CMD 31010) | PX4 v1.15.4, EKF2, Commander |
| Node ROS `ros/` (perch_node, tag_camera_sim, eval_node) | ROS 2 Jazzy, px4_msgs, Micro XRCE-DDS Agent |

## 3. Tái lập từng kết quả

Huấn luyện chạy qua hàng đợi GPU (C4). Mỗi file `/home/sup/data/perchpilot/queue/NN_tên.job` chứa một dòng lệnh. Runner chạy dưới `systemd-inhibit` để laptop không suspend giữa chừng:

```bash
IDLE_MAX=21600 setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch --who=perchpilot \
  --why="GPU training queue" --mode=block bash scripts/gpu_queue.sh &
```

Ngân sách `--minutes` của `train.py` đo bằng `time.monotonic()` nên thời gian suspend không bị tính (sửa ngày 2026-11-19).

### 3.1 Q1 giai đoạn 1: có trễ PX4, state đặc quyền (ĐÃ CHẠY, 3 seed)

```bash
# G1: state PPO, không trễ (checkpoint khởi động ấm cho P1), tạo bằng script tuần chạy thử,
# kết quả gốc results/g1_state_ppo_summary.json. Kiểm tái lập bằng train.py: job 50_p0_g1repro20
$PY scripts/train_pilot.py --minutes 20 --out /home/sup/data/perchpilot/runs/g1_state_ppo
# P1: trễ 40–60 ms, TW 1,9–2,6, đế 0,227 m; seed 0, 1, 2
$PY scripts/train.py --name p1b_s0 --seed 0 --phase P1 --minutes 25 --tw 1.9,2.6 --pad_offset 0.227 \
  --calib --hist --asym --init /home/sup/data/perchpilot/runs/g1_state_ppo/ckpt.pt --critic_warmup 50 \
  --gate --ent_delay 0.003 --lr_max_delay 1e-4 --std_max 0.35
# Chấm lại bằng mã metric hiện tại (4096 lượt, seed eval 12345)
$PY scripts/eval_ckpt.py /home/sup/data/perchpilot/runs/p1b_s0
```

### 3.2 Q1 giai đoạn 2: camera, nhiễu odometry, ρ (ĐÃ CHẠY, 3 nhóm × 3 seed)

```bash
$PY scripts/train.py --name p2_perc_s0 --seed 0 --perc --phase P2 --minutes 30 --tw 1.9,2.6 --pad_offset 0.227 \
  --calib --hist --asym --v2 --init /home/sup/data/perchpilot/runs/p1b_s0/best.pt --critic_warmup 30 \
  --ent 0.003 --lr 1e-4 --lr_max 1e-4 --std_max 0.35
# Ablation: p2_norho_sN = bỏ --perc, thêm --no_rho; p2_percnorho_sN = giữ --perc, thêm --no_rho
```

### 3.3 Q1 với mô hình actuator 4 motor (họ chính dùng để triển khai, ĐÃ CHẠY 3 seed)

```bash
# p2y_perc_sN: mô hình 4 motor bão hoà + airmode, không có lệnh yaw (giống lúc triển khai)
$PY scripts/train.py --name p2y_perc_s0 --seed 0 --perc --phase P2Y --minutes 35 --tw 1.9,2.6 --pad_offset 0.227 \
  --calib --hist --asym --v2 --motor --yaw_scale 0 --init /home/sup/data/perchpilot/runs/p2_perc_s0/best.pt \
  --critic_warmup 30 --ent 0.003 --lr 1e-4 --lr_max 1e-4 --std_max 0.35
# Chấm lại với cấu hình khác: eval_ckpt.py RUN --set yaw_scale=0 --set motor_model=true --tag noyaw
```

### 3.4 Thí nghiệm P6: độ nhạy, đối chứng, phân loại lỗi (ĐÃ CHẠY)

```bash
$PY scripts/sensitivity.py /home/sup/data/perchpilot/runs/p2y_perc_s0 --n 4096 --seed 12345   # odo ×0/1/2/4, trễ 30–90 ms
$PY scripts/baselines_q1.py --run /home/sup/data/perchpilot/runs/p2_perc_s0 --n 4096 --seed 12345  # B0 bay thẳng, B6 bay treo
```

### 3.5 Q2: xác nhận bám (ĐÃ CHẠY v4–v7 quy mô đầy đủ)

Giao thức v4 (đặc tả v1.2) có 4 split: train, val, dev_ood, test.
- Ngưỡng được khoá bằng cận trên Wilson trên dev_ood, ghi hash vào file lock, rồi chấm test đúng một lần.
- False-safe = số lượt rơi (kể cả mất tiếp xúc sau lệnh OFF) chia cho số lệnh OFF.

| Phiên bản | Cờ `--cfg` của dataset | Ý nghĩa |
|---|---|---|
| v4 | (không có) | Cảm biến danh định, 11 kênh |
| v5 | `sensor_dr=1` | Ngẫu nhiên hoá load cell, IMU, rung |
| v6 | `sensor_dr=1,slip_sensor=1` | Thêm cảm biến trượt quang (C5 v2, 13 kênh), đối chứng B4e |
| v7 | `sensor_dr=1,slip_sensor=1,imu_hold=1` | Thêm IMU giữ mẫu 100 Hz như PX4 SITL |

```bash
D=/home/sup/data/perchpilot/q2data/v7_full; R=/home/sup/data/perchpilot/runs
$PY q2/dataset.py --protocol v4 --cfg sensor_dr=1,slip_sensor=1,imu_hold=1 --out $D --seed 0 --chunk 2048 \
  --n_train 200000 --n_val 20000 --n_dev_ood 20000 --n_test 20000 --device cuda
for s in 0 1 2 3 4; do $PY q2/train_belief.py --data $D --tag v7_full_s$s --out_root $R --seed $s --epochs 16 --bs 512 --device cuda; done
$PY q2/decide_v4.py --data $D --tag full --belief $R/q2_belief_v7_full_s{0,1,2,3,4} --device cuda   # → results/q2_v7_full{,_lock}.json
$PY q2/sitl_external.py && $PY q2/eval_sitl_external.py --data /home/sup/data/perchpilot/q2data/sitl_ext ...  # tập test ngoài từ log SITL
```

Các phiên bản khác (v4, v5, v6) chạy cùng lệnh, chỉ đổi cờ `--cfg` của dataset theo bảng trên.

### 3.6 Bảng, hình và video

```bash
MPLCONFIGDIR=/tmp/mpl $PY scripts/make_figures.py     # → results/TABLES.md, results/fig/*.png
$PY scripts/test_make_figures.py && $PY scripts/test_sensitivity.py
$PY scripts/make_video.py --montage --fps 30 --speed 0.4 --dpi 100 --decimate 2   # video dựng từ log SITL
```

### 3.7 Gazebo + PX4 SITL vòng kín (ĐÃ CHẠY)

Stack SITL **cô lập** với các dự án khác trên máy:
- `GZ_PARTITION=perchpilot`;
- PX4 instance 7;
- agent XRCE-DDS cổng UDP 8899;
- `ROS_DOMAIN_ID=77`, namespace `/px4_7`.

Chỉ chạy 1 SITL một lúc. Các mặc định của `run_sitl_eval.sh` là cấu hình đánh giá chính:
- HoldPad gắn theo diện tích càng (C2 v1.1);
- load cell đo tổng lực;
- có cảm biến trượt;
- `MC_AIRMODE=1`, không lệnh yaw, giữ vận tốc tấm trong PERCH;
- DEPART kiểu push-off.

```bash
(cd gz/plugins/hold_pad && cmake -S . -B build && cmake --build build)   # plugin HoldPad + test C++
# Lô chính Q1 + Q3 (một vật liệu bám chắc):
scripts/run_sitl_eval.sh --name main_p2y_v15 --policy-run /home/sup/data/perchpilot/runs/p2y_perc_s0 \
  --trials 10 --eval-seed 12345 --speeds '1.5' --rest-s 5 --timeout-s 45
# Lô Q2 (lưới 3 × 3 vật liệu, luật B4e với ngưỡng từ q2_v6_full_lock.json ở trần 0,02):
PERCH_NODE_ARGS='--verify-rule fsm_b4e --q2-slip-off 0.5467 9.2340 0.7578' scripts/run_sitl_eval.sh \
  --name q2_b4e --policy-run /home/sup/data/perchpilot/runs/p2y_perc_s0 --trials 27 --speeds '0' \
  --mu '0.2;0.5;0.9' --f-hold-n '10.13;24.3;60.75' --q2 --rest-s 10 --timeout-s 50
# D-thr: PERCH_NODE_ARGS='--verify-rule dthr --q2-belief /home/sup/data/perchpilot/runs/q2_belief_v7_full_s0 --q2-tau 0.44 --q2-s 1.01'
scripts/sitl_down.sh          # chỉ dừng các PID do sitl_up.sh ghi lại
$PY ros/q2_pool.py ...        # gộp các lô cùng cấu hình: Wilson, Newcombe, Fisher
```

Tốc độ xe viết `'1.5'`, không viết `'1,5'`, vì dấu phẩy là dấu phân tách lưới. Test offline: `pytest ros/` sau khi source ROS (`/opt/ros/jazzy/setup.bash`, `~/ws_px4/install/setup.bash`), dùng python của venv. Ghi chú gỡ lỗi chi tiết nằm ở `ros/README_perch_node.md`.

[Dashboard giám sát ROS 2 và replay SITL](ros/README_perch_node.md#dashboard-giám-sát-p7) hiển thị pha bay, telemetry, quyết định Q2 và kết quả chấm tại `http://127.0.0.1:8765`.

## 4. Ràng buộc đánh giá (R1–R6)

| Mã | Ràng buộc |
|---|---|
| R1 | Lúc test không dùng truth pose. Node chính sách chỉ đọc `/fmu/out/*` (EKF2) và `/perch/tag_pose`. Chỉ `eval_node` đọc truth để chấm |
| R2 | Không dùng camera depth |
| R3 | μ, F_hold và nhãn chỉ dùng khi huấn luyện và chấm điểm, không bao giờ là đầu vào |
| R4 | Không dùng DetachableJoint khi chấm điểm (dùng HoldPad) |
| R5 | SITL cô lập |
| R6 | Báo cáo seed và khoảng Wilson 95% |

Giới hạn cần ghi rõ trong báo cáo:
- Camera là **bộ giả lập phép đo**, chưa phải AprilTag render thật.
- Mô hình rung do trượt của Q2 là **giả định chưa hiệu chuẩn**.
