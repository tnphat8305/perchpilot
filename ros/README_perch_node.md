# perch_node — chính sách đậu chạy vòng kín với PX4 SITL (P5 sớm)

Ngày 2026-11-19. Trạng thái: **đã test offline và Gazebo headless, CHƯA chạy với SITL.**

## 1. Các file

| File | Nội dung |
|---|---|
| `perch_node.py` | Node ROS 2 (rclpy). Đọc odometry/trạng thái PX4 và tag. Phát lệnh PX4 và `/perchpilot/diag`. |
| `perch_core.py` | Lõi không phụ thuộc ROS: đổi hệ trục, quan sát v1/v2, ánh xạ hành động → PX4, nạp chính sách (TorchScript hoặc numpy), máy pha, thống kê thời gian. |
| `plate_perception.py` | Biến đổi pose camera và Kalman vận tốc hằng. |
| `tag_camera_sim.py` | Đọc pose Gazebo, tổng hợp phép đo của camera và phát `/perch/tag_pose`. |
| `export_policy.py` | Công cụ tạm: đổi `ckpt.pt` (ActorCritic của `rl/ppo.py`) sang `policy.ts` + `policy.json` + `policy.npz` theo C1. |
| `test_perch_node.py` | 34 test offline + 1 test tuỳ chọn với chính sách thật. |

## 2. Python nào chạy được node

- `python3` hệ thống (thực ra là `/usr/bin/python3`, 3.12.3) **không có torch**.
- venv `/home/sup/data/venvs/occwm` cũng là CPython 3.12.3, có torch 2.15 và numpy 1.26.4 (trùng bản numpy apt mà ROS Jazzy dùng).
- Vì vậy chỉ cần source ROS rồi gọi python của venv. `PYTHONPATH` và `LD_LIBRARY_PATH` do `setup.bash` đặt là đủ. Đã thử: `import rclpy, px4_msgs, std_msgs, torch` đều được.

```bash
source /opt/ros/jazzy/setup.bash && source ~/ws_px4/install/setup.bash
export ROS_DOMAIN_ID=77
/home/sup/data/venvs/occwm/bin/python perchpilot/ros/perch_node.py --run RUN_DIR
```

**Dự phòng không cần torch:** `--backend numpy` đọc `RUN/policy.npz`. Đã chạy được bằng `/usr/bin/python3`.
- `--backend auto` (mặc định) thử torch trước. Không có torch thì tự chuyển sang numpy.
- Đề xuất mở rộng C1 (file này không sửa `docs/CONTRACTS.md`): thêm file tuỳ chọn `policy.npz` với các khoá:
  - `norm_mean`, `norm_var`;
  - `n_layers`, `W0..W{L-1}` (dạng [out, in]), `b0..b{L-1}`;
  - kích hoạt ELU giữa các lớp; chuẩn hoá `(x−mean)/(sqrt(var)+1e-6)` cắt về [−10, 10]; đầu ra cắt về [−1, 1].
- Hai backend cho cùng kết quả, sai khác < 1e-5 (có test).

## 3. Xuất chính sách (cho tới khi `train_pilot.py` tự ghi `policy.ts`)

```bash
/home/sup/data/venvs/occwm/bin/python perchpilot/ros/export_policy.py \
    --ckpt /home/sup/data/perchpilot/runs/g1_state_ppo/ckpt.pt --out /home/sup/data/perchpilot/export/g1_state_ppo
```

- Trường `git` trong `policy.json` là hash lúc **xuất**, không phải lúc huấn luyện. `seed` mặc định là 0.
- `delay_steps_trained` lấy từ `summary.json["calib"]` nếu có.
- Torch 2.15 cảnh báo `torch.jit.script/load` đã bị deprecated, nhưng vẫn chạy được.

## 4. Chạy với SITL (sau khi bật `scripts/sitl_up.sh`)

Source ROS và đặt `ROS_DOMAIN_ID=77` ở cả hai terminal. Camera dùng partition `perchpilot` và drone `x500_perch_7` theo launcher mặc định.

```bash
# Terminal camera
source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=77
GZ_PARTITION=perchpilot /home/sup/data/venvs/occwm/bin/python perchpilot/ros/tag_camera_sim.py

# Terminal chính sách
source /opt/ros/jazzy/setup.bash && source ~/ws_px4/install/setup.bash && export ROS_DOMAIN_ID=77
/home/sup/data/venvs/occwm/bin/python perchpilot/ros/perch_node.py --run RUN_DIR \
    --thrust-model x500_perch_gz --log-dir /home/sup/data/perchpilot/sitl/perch_node_run1
```

- Khi chọn `PX4_SIM_MODEL=gz_x500`, chạy camera với `--drone-model x500_7` hoặc export cùng `PX4_SIM_MODEL` trong terminal camera. Chọn `--thrust-model x500_gz` ở node chính sách. Xem [bảng thrust](../gz/models/x500_perch/README.md).
- Node ở WAIT đến khi có ước lượng tấm. Launcher dùng pose `1.55,0,0.25,0,0,0` để camera thấy tấm trước khi arm. Khi đổi pose bằng `PX4_GZ_MODEL_POSE`, kiểm tra `ros2 topic echo /perch/tag_pose`.
- Node stream setpoint 1 s trong APPROACH rồi gửi lệnh offboard và arm.
- Mặc định `--thrust-model linear` lấy hover thrust từ `--hover-thrust`, rồi `px4-param --instance 7 show MPC_THR_HOVER`, cuối cùng là 0,60.
- `--log-dir` ghi `ticks.jsonl` và `summary.json` với chu kỳ vòng lặp, jitter và thời gian suy luận. Có thể xem `ros2 topic echo /perchpilot/diag`.
- `--dry-run --duration 20` chạy pha PERCH trên cảnh tổng hợp trong ROS domain riêng, không cần PX4/Gazebo.

## 5. Quy ước hệ trục (chỗ dễ sai nhất)

| Đại lượng | PX4 (vào/ra) | Sim / Gazebo / lõi node | Đổi |
|---|---|---|---|
| Vị trí, vận tốc | NED | ENU | `(x,y,z)_enu = (y, x, −z)_ned` |
| Trục thân | FRD | FLU (z = trục lực đẩy) | `(x, −y, −z)` |
| Quaternion | `q` quay FRD→NED, [w,x,y,z] | R quay FLU→ENU | `R_enu_flu = T · R(q) · F` |
| Body rate | (p, q, r) FRD | ω FLU | `(p, −q, −r)` |
| Lực đẩy | `thrust_body = [0, 0, −thr]` | dọc +z thân | dấu âm vì z FRD hướng xuống |
| Yaw | từ Bắc, theo chiều kim đồng hồ | từ Đông, ngược chiều kim đồng hồ | `ψ_ned = π/2 − ψ_enu` |

Trong đó `T = [[0,1,0],[1,0,0],[0,0,−1]]` và `F = diag(1,−1,−1)`; cả hai đều là nghịch đảo của chính nó.

**Tấm trong Gazebo:**
- Pose của tấm trong hệ cart là `(−0.8, 0, 0.85)`, pitch −1.1345 rad.
- Suy ra pháp tuyến `n = R_cart·(−sin65°, 0, cos65°)` và trục ngang `u = R_cart[:,1]`, giống hệt `plate_frame()` của sim.
- Tâm dùng cho quan sát là **tâm mặt tấm**, tức tâm hộp + 0,01·n. Trong sim, `c` là điểm mà `d = (pad − c)·n = 0` lúc chạm.

## 6. Bộ giả lập nhận thức tấm và quy tắc R1

- `tag_camera_sim.py` đọc pose drone và cart cùng một frame trên `/world/perch_world/dynamic_pose/info`. Camera đặt tại gốc `base_link`, lệch +0,24 m theo z so với gốc model x500, chúc xuống 15°. Hệ `camera` là FLU, x tiến, y trái, z lên.
- Camera phát `geometry_msgs/PoseStamped` trên `/perch/tag_pose` với timestamp ROS, 30 Hz khi thấy tag. Pose biểu diễn tâm mặt tấm và hướng tấm trong camera. Cổng nhìn theo sim v2 với FOV 82,96° × 52,90°, khoảng cách <6 m và góc giữa pháp tuyến mặt trước và hướng nhìn từ tấm về camera <70°.
- Độ lệch chuẩn nhiễu vị trí là `0,01 + 0,01·d` m với `d` tính bằng m. Độ lệch chuẩn nhiễu cộng vào pháp tuyến là 2° quy đổi sang radian trước khi chuẩn hoá.
- `perch_node.py --perception tag` là mặc định. Node nhận tag, `vehicle_odometry` và `vehicle_status`. Node ghép pose camera với pose EKF2 mới nhất trước khi cập nhật KF vận tốc hằng có `q_acc=0,5` và EMA hướng tấm. Node dùng tính hình học để phát hiện tiếp xúc.
- Một phép đo có hiệu lực nhìn thấy trong một chu kỳ 1/30 s. Sau đó `T_loss` tăng theo thời gian không có tag, giới hạn 2 s như sim v2. PerchRL Eq. 12 cho `ρ = exp(−2·T_loss)`, bằng 1 khi đang thấy tag.
- Chính sách v2 dùng hậu tố `[rho, visible, t_loss_norm]` sau lịch sử hành động. `export_policy.py` ghi thứ tự này trong `obs_layout`. Định dạng v1 tiếp tục được nạp.

## 7. Máy pha

| Pha | Lệnh gửi PX4 | Chuyển pha |
|---|---|---|
| WAIT | setpoint vị trí tại chỗ | ước lượng tấm hợp lệ → APPROACH |
| APPROACH | offboard position tới `c + 3·n`, kèm vận tốc tấm làm feed-forward, yaw nhìn về tấm; tự gửi lệnh offboard + arm | sai số < 0,5 m, vận tốc tương đối < 0,6 m/s, thấy tấm liên tục ≥ 0,3 s → PERCH; quá 90 s → ABORT |
| PERCH | 100 Hz: quan sát → chính sách → `VehicleRatesSetpoint` + `OffboardControlMode(body_rate)` | có tiếp xúc theo hình học → VERIFY (`d ≤ 2 cm` và trong mặt tấm); mất tấm > 1 s, đủ 600 bước, lật, quá thấp hoặc bị disarm → ABORT |
| VERIFY | placeholder: giữ thrust cuối, rate = 0, ghi log | sau 0,5 s: còn tiếp xúc → REST, không thì → ABORT |
| REST | 31010 p1=1, disarm thường; còn armed sau 1 s thì disarm cưỡng bức 21196; vẫn stream setpoint vị trí tại chỗ | hết `--rest-s` → DEPART |
| DEPART | DO_SET_MODE + arm; bị từ chối 2 lần thì arm cưỡng bức; armed rồi thì gửi 31010 p1=0 và `/perch/hold_cmd`=0, setpoint = vị trí + 2 m dọc n | tới setpoint và cách tấm ≥ 2 m → DONE |
| ABORT | `/perch/hold_cmd`=0, setpoint lùi 1,5 m theo n ngang và lên 1 m | sau 3 s → APPROACH (tối đa `--max-attempts`), hết lượt → DONE |

- Thứ tự lệnh ở DEPART (arm trước, tắt perch mode sau) theo đúng lượt G2-ride đã chạy được. Đặc tả ghi ngược lại, tức tắt perch mode trước rồi mới arm. Làm theo đặc tả có thể bị check GPS "đứng yên" chặn arm khi xe đang chạy.
- EKF cũ quá 0,2 s thì node không publish gì, để PX4 tự vào failsafe mất offboard.

**Quan sát trong PERCH** (giống env lúc reset):
- `prev_act` khởi tạo bằng hành động hover `2/TW − 1`;
- `t_norm = k/600`, với k đếm từ lúc vào PERCH;
- lịch sử hành động theo `act_buf[:,1:]` nếu cấu hình huấn luyện có bật. Mục mới nhất đứng trước, trước ba giá trị camera của v2.

## 8. Đã test và chưa test

**ĐÃ CHẠY** (`/home/sup/data/venvs/occwm/bin/python -m pytest -q perchpilot/ros/test_perch_node.py`): 34 passed, 1 skipped (test tuỳ chọn).

| Nhóm | Nội dung |
|---|---|
| Hệ trục | round-trip vector/quaternion/yaw; các tư thế đã biết (PX4 identity, yaw 90°, pitch lên); rate FRD↔FLU nhất quán với phép quay (tích phân 1 bước); vận tốc theo hệ BODY_FRD |
| Quan sát | `build_obs` == `QuadPerchEnv.observe()` (CPU, num_envs=1, obs_noise=0), sai số < 1e-5, 100 trạng thái ngẫu nhiên. PX4 msg được dựng **độc lập** bằng đại số quaternion của sim, không dùng ma trận của lõi |
| Vòng kín lõi ↔ env | env → PX4 msg → PhaseMachine → `VehicleRatesSetpoint` → hành động → `env.step`, 900 bước có reset, cả khi có và không có `act_history` (obs 52 chiều, trễ 4–6); obs khớp từng bước |
| Ánh xạ hành động | dấu roll/pitch/yaw, thrust âm theo z FRD, điểm hover, bão hoà, hàm ngược; mô hình động cơ x500 |
| Chính sách | TorchScript và numpy cho cùng kết quả (28 và 52 chiều); kiểm tra `policy.json`; suy luận CPU với lưới 256-256-128: torch p50 0,18 ms / p95 0,23 ms, numpy p50 0,21 ms / p95 0,23 ms |
| Nhận thức | hình học tấm từ SDF; FOV; đúng 30 frame/s; KF bám; round-trip camera/thân/ENU, FOV khớp sim v2 trên 1.024 trạng thái, ρ và parse `gz.msgs.Pose_V` |
| Máy pha | đủ chuỗi WAIT→…→DONE (gồm disarm/arm cưỡng bức); nhánh ABORT; không có EKF thì không publish |
| R1 tĩnh | node chỉ subscribe odometry, status và tag pose |

**Test tuỳ chọn ĐÃ CHẠY** với chính sách thật `g1_state_ppo` (xuất ra scratchpad):

```bash
PERCH_RUN=<export_dir> PERCH_CKPT=/home/sup/data/perchpilot/runs/g1_state_ppo/ckpt.pt pytest -k real_policy -s
```

- TorchScript khớp `act_deterministic` (sai số < 1e-5).
- Đi qua đường node thì thành công 30/30, đường trực tiếp cũng 30/30.

**Dry-run node ROS của P5 sớm** (domain 93, localhost, 20 s, torch):
- 2000 tick trong 20 s, 600 phép đo (đúng 30 Hz);
- chu kỳ p50 10,00 ms / p95 12,2 ms;
- jitter p50 0,11 ms / p95 3,1 ms;
- suy luận p50 0,44 ms / p95 0,62 ms;
- tính một tick p50 1,1 ms / p95 3,7 ms.

`/usr/bin/python3` + numpy cho kết quả tương tự. Jitter tăng khi máy bận: khi chạy kèm `ros2 topic echo`, p95 lên 5 ms.

**Gazebo headless ĐÃ CHẠY.** Spawn `x500_perch` và nhận `/perch/tag_pose` từ world.

**CHƯA CHẠY.** Vòng kín PX4 SITL, QoS uXRCE, lệnh 31010, arm/disarm, hiệu chỉnh hover và bridge HoldPad trong P5.

## 9. Vấn đề mở

1. **Chọn checkpoint đúng TW.** G1 huấn luyện với TW 2,6–3,4. Dùng các lượt P1/P2 huấn luyện với TW 1,9–2,6 và ghi TW danh nghĩa 2,26 trong metadata. Đo thrust lúc treo và kiểm tra bão hoà cùng rate setpoint.
2. **Mất tag và sai thời gian.** Tag có thể ra khỏi FOV gần touchdown. KF dự báo tối đa theo `--perch-lost-s` (mặc định 1 s). Ghép pose dùng mẫu EKF2 mới nhất còn hạn 0,2 s, nên độ lệch thời điểm và sai số thái độ ảnh hưởng phép đo khi drone quay nhanh.
3. **Căn hình học và tích hợp HoldPad.** Chốt `pad_offset` giữa sim, node và HoldPad. HoldPad khởi tạo tắt, pha tiếp xúc P5 cần đường gửi `hold_cmd=1` và bridge lệnh nhả. VERIFY/REST hiện là khung.
4. **Jitter.** Dry-run P5 sớm từng có jitter p95 ≈3 ms khi máy đang tải nặng. Đo lại với uXRCE và camera đang chạy.

## 10. Gỡ lỗi vòng kín SITL đầu tiên (2026-11-19)

ĐO = đo trên log SITL (ulog PX4, truth Gazebo, `node_ticks.jsonl`). ƯL = ước lượng bằng sim CPU sửa tạm trong scratch, không sửa `perchpilot/sim/`.

| # | Nguyên nhân | Bằng chứng | Sửa |
|---|---|---|---|
| 1 | Commander PX4 chỉ cập nhật `vehicle_control_mode` 2 Hz khi đổi offboard position → body-rate. 0,37 s đầu PERCH PX4 vẫn bay position, policy chạy "mù" | ĐO, ulog `p5_smoke1` | PERCH chờ `vehicle_control_mode` báo rate-only, trong lúc chờ gửi level-hold + setpoint vị trí, `k` chưa chạy. Đổi về position thì gửi level-hold rate cho tới khi PX4 đổi |
| 2 | `MC_AIRMODE=0`: thrust 0 thì không có mô-men, lệnh "cắt thrust + pitch 6 rad/s" của policy không quay được máy | ĐO, ω ≈ 0 suốt 0,4 s | `run_sitl_eval.sh` đặt `MC_AIRMODE=1` qua `--px4-param` |
| 3 | Abort "too low" theo z EKF, nhưng gốc z EKF lệch −0,24 m / +1,4 m tuỳ phiên | ĐO | Ngưỡng theo tâm tấm: `min_below_plate` 1 m |
| 4 | Điểm bàn giao `c + 3n` nằm ở mép phân bố reset của sim, drone còn bay 0,6 m/s | ĐO | Điểm `c − 4,0·f + 1,0·z`, `verr < 0,3`, yaw < 0,25 rad, giữ 0,3 s |
| 5 | Ghép tag với EKF mới nhất và stamp lúc publish: sai tâm tấm rms 0,23–0,28 m khi quay nhanh | ĐO | Stamp theo thời điểm mẫu pose; node ghép với mẫu EKF gần stamp nhất. Còn rms 4–7 cm |
| 6 | KF tấm `q_acc=0,5`: `veh_speed` lệch +0,32 m/s khi xe đứng yên | ĐO | Mặc định 0,1. Vẫn còn lệch do EKF trôi khi bay gắt |
| 7 | **Chưa sửa được.** Policy đòi thrust tối đa và rate lớn cùng lúc. Trên x500, airmode lấy bớt collective để có mô-men: lệnh 2,2 g chỉ ra 1,2–1,4 g. Sim cho rate bám bậc nhất, độc lập với thrust | ĐO thrust (IMU và truth). ƯL: thay actuator sim v2 bằng phân bổ 4 motor kiểu PX4 airmode thì `p2_perc_s0` từ điểm bàn giao còn 0,6% (lý tưởng 92%), danh định 30% (79%) | Cần huấn luyện lại với mô hình motor/phân bổ |

Đo trễ (ĐO): ROS → PX4 ≈ 0 ms; PX4 rate loop có trễ 24 ms + τ 30–45 ms (pitch), 24–32 ms + τ 50–70 ms (roll). Gia tốc góc tối đa pitch ~90–115 rad/s², roll ~60. Odometry chậm hơn truth ~10–15 ms. Tổng khớp dải đã huấn luyện (4–6 bước + τ 35–70 ms). RTF Gazebo 0,97–1,00.

Công cụ: `sitl_obs_check.py RUN_DIR` so quan sát của node với quan sát lý tưởng tính từ truth, theo từng thành phần. Cùng lúc báo sai số EKF và tấm, trễ rate, RTF.

## 11. Vòng kín với chính sách `p2m_perc_s0` (mô hình motor, 2026-11-20)

Lô chạy với `MC_AIRMODE=1` qua `run_sitl_eval.sh`. Cấu hình cuối có thêm ba tuỳ chọn:

```bash
PERCH_PAD_FOOTPRINT="0.125 0.132" PERCH_NODE_ARGS='--yaw-scale 0 --freeze-plate-v' perchpilot/scripts/run_sitl_eval.sh ...
```

| Bước | Lỗi tìm ra | Bằng chứng | Sửa |
|---|---|---|---|
| `p5m_dbg_c1`: 0/4 chạm | Policy giữ lệnh yaw ±3 rad/s suốt PERCH. PX4 trộn yaw sau cùng và hạ collective: lệnh 2,2 g chỉ ra 1,45 g | ĐO: IMU và motor trong ulog; `vehicle_torque_setpoint` yaw 0,3–0,9. ƯL: trong sim v2 có motor, bỏ yaw làm thành công tăng 80→89% (danh định), 92→97% (bàn giao) | `--yaw-scale 0`: yaw-rate gửi PX4 bằng 0, quan sát và lịch sử vẫn giữ hành động gốc của policy. Thrust lúc lệnh tối đa lên 16–18 m/s² |
| `p5m_dbg_c2`: 0/4 chạm | Càng đáp chạm tấm trước đế. Lúc chạm, trục thân lệch pháp tuyến 15–22°, nên đầu càng (x ±0,125, y ±0,132) chạm khi điểm đế còn cách tấm 3–6 cm. HoldPad chỉ gắn khi điểm đế cách ≤ 1 cm | ĐO: truth Gazebo và tên collision `base_link_collision_3/4`. ƯL: 99% số lượt thành công trong sim cũng lệch 15–20°, nên trong Gazebo đều chạm càng trước. Tiêu chí của sim (điểm đế, lệch < 25°) không tính hình học càng | Tuỳ chọn HoldPad `footprint_half_x/y`: gắn khi góc thấp nhất của hình chữ nhật càng chạm. Bật bằng `PERCH_PAD_FOOTPRINT`. Mặc định vẫn là điểm đế theo C2 |
| `p5m_dbg_c3`: 4/4 chạm, nhưng ngoài đường bao | v_n lúc chạm 1,63–1,81 m/s (ngưỡng 1,5). KF tấm, ghép với EKF đang trôi khi bay gắt, thấy tấm "chuyển động" 0,2 m/s. `rel_v_n` lệch +0,3 m/s gần lúc chạm, nên policy phanh thiếu | ĐO: `sitl_obs_check.py` (vận tốc tấm lệch +0,22 m/s theo x, rel_v n lệch +0,17 trung bình) | `--freeze-plate-v`: trong PERCH chỉ cập nhật vị trí tấm, giữ vận tốc ước lượng lúc treo. Hiệp phương sai dùng dạng Joseph |
| `p5m_dbg_c4`: **4/4 thành công và giữ được 5 s** | — | v_n 1,23–1,24, lệch 15–20°, sai vị trí 7–9 cm | — |

So mô hình motor của sim với SITL (ĐO). Phát lại open-loop các setpoint ghi trong ulog qua `QuadPerchEnv._allocate`:
- Khi tắt yaw, lệnh thrust tối đa 21 m/s² cho 16,4 ở sim và 17,8 ở SITL.
- Khi bật yaw: 15,9 ở sim và 14,3 ở SITL. Sim thiếu cơ chế PX4 hạ collective để nhường chỗ cho yaw.
- Lệnh thrust ≈ 0: sim cho 5,4–7,9, SITL 4,3.
- Sai số rate: sim 0,25/0,33 rad/s (roll/pitch).
- PX4 phân bổ trong không gian u (ω tuyến tính) chứ không trong không gian lực. Sim chưa mô hình hoá điều này.

**Lô 10 lượt (ĐÃ CHẠY, cả hai lô dừng sớm vì phục hồi giữa các lượt thất bại):**

| Lô | Xe | Đã chạy | Chạm đạt và giữ khi cắt motor | Ghi chú |
|---|---|---|---|---|
| `sitl_p5m_v0` | 0 m/s | 6 | 4/6 | 2 lượt trượt ngang: đế lệch u 0,24–0,30 m, ra ngoài mép tấm. Ước lượng ngang trôi khoảng 13 cm khi mất tag gần lúc chạm. Sau lượt 6, PX4 không cho arm lại trên tấm |
| `sitl_p5m_v15` | 1,5 m/s | 2 | 2/2 | DEPART quá 15 s thì node kẹt ở DONE. Đã sửa: thử DEPART lại hoặc bay về điểm chờ |
| `sitl_p5m_v15b` | 1,5 m/s | 1 | 1/1 | Sau khi rời tấm trên xe đang chạy, máy bay báo "Attitude failure (roll)" và không về được điểm chờ |

## 12. Phục hồi giữa các lượt và lô chính (2026-11-20)

**Vì sao PX4 từ chối arm lại (ĐO, ulog `17_51_56`).** Pha DEPART cũ arm ở offboard POSITION khi drone còn nghiêng 43° trên tấm:
- `takeoff_status` đang ở `ready_for_takeoff`: `thrust_sp` = 0 rồi mới tăng dần từ 0 lên 0,66 trong 0,4 s, và tilt bị giới hạn 0,21 rad.
- Sau khi nhả, drone rơi 1,25 m xuống đất trong 0,8 s (truth) rồi lật.
- EKF2 làm "GPS emergency yaw reset", bật `emergency_yaw_reset_mag_stopped`, nên `cs_mag_fault` bị chốt.
- Từ đó commander báo "Compass needs calibration" và mọi lệnh arm sau bị từ chối với "Resolve system health failures first".

Bộ phát hiện lỗi tư thế (`FD_FAIL_R/P` = 60°) chỉ chạy khi chế độ điều khiển có vòng attitude. Nó là nguồn của "Attitude failure (roll)" ở lượt 1,5 m/s cũ.

**Sửa (không đổi tham số PX4):** `--depart-pushoff`, mặc định trong `run_sitl_eval.sh`. Trình tự:
1. Arm ở offboard body-rate, thrust idle 0,05, rate 0. Ở chế độ này không có ramp cất cánh và không có failure detector.
2. Chờ spool 1 s.
3. Nhả HoldPad và tắt perch mode.
4. Đẩy 1,5 g dọc trục z thân (gần trùng pháp tuyến tấm) trong 0,25 s, rate 0.
5. Level-hold có hãm v_z trong 0,6 s.
6. Chuyển sang offboard position, đích cách 2 m theo pháp tuyến nằm ngang và cao thêm 1 m.

**Fallback:** nếu một lượt không về được điểm chờ, `eval_node` thoát với mã 3. `run_sitl_eval.sh` khởi động lại PX4 và Gazebo rồi chạy tiếp bằng `--continue`, tối đa `PERCH_MAX_RESTARTS` lần.
- Lượt đó vẫn được tính, với `recovered=false`, và ghi số lần khởi động lại ở `sitl_restarts`.
- Metric mới: `rearm` là tỉ lệ arm lại được trong DEPART, mẫu số là số lượt đã cắt motor trên tấm. `cycle_complete` là trọn chu kỳ thành công, arm lại và về điểm chờ.

**Lô chính (ĐÃ CHẠY):** `p2m_perc_s0`, C2 v1.1, rest 5 s, `--yaw-scale 0 --freeze-plate-v --depart-pushoff`.

| Lô | Xe | n | Chạm đạt | Giữ khi cắt motor | False-safe | Arm lại | Trọn chu kỳ | SITL restart |
|---|---|---:|---|---|---|---|---|---:|
| `sitl_main_v0` | 0 m/s | 20 | 20/20 [0,84; 1] | 19/20 [0,76; 0,99] | 0/19 [0; 0,17] | 19/19 [0,83; 1] | 19/20 [0,76; 0,99] | 0 |
| `sitl_main_v15` | 1,5 m/s | 20 | 20/20 [0,84; 1] | 20/20 [0,84; 1] | 0/20 [0; 0,16] | 20/20 [0,84; 1] | 20/20 [0,84; 1] | 0 |
| `sitl_main_v3` | 3 m/s | 10 | 8/10 [0,49; 0,94] | 8/10 [0,49; 0,94] | 0/8 [0; 0,32] | 8/8 [0,68; 1] | 8/10 [0,49; 0,94] | 0 |

Lượt hỏng duy nhất là lượt 5 ở 0 m/s, kết quả timeout:
- Lượt này chạm đạt (v_n 1,2 m/s), nhưng PX4 không disarm. Lệnh disarm thường bị từ chối ("not landed"), còn lệnh disarm cưỡng bức 21196 chỉ gửi một lần và có vẻ bị mất trên DDS best-effort.
- Đã sửa sau lô 0 m/s: lặp lại lệnh disarm cưỡng bức mỗi 1 s khi còn armed. Lô 1,5 m/s chạy với code cũ (bản vá được tạm cất cho tới khi lô xong), lô 3 m/s chạy với bản đã sửa.

Ở 3 m/s có 2 lượt chạm ngoài đường bao vì v_t = 1,03 và 1,59 m/s (ngưỡng 1,0), trong khi lệch góc 7–8° và v_n 0,7–1,0 m/s vẫn đạt.

## 13. Q2 trong SITL (2026-11-20)

`--verify-rule {none, dthr, fsm_b4b, probe}` (`perch_q2.py`). Pha VERIFY chạy macro RELAX/PULL của `sim/contact_phase.py` (v2) bằng offboard body-rate với rate 0. Node lấy mẫu C5 ở 200 Hz bằng timer riêng, gồm:
- `sensor_combined` FRD đổi sang FLU;
- `/perch/hold_sensors` cho load cell và contact;
- thrust lệnh / (TW danh nghĩa · g);
- phase_id và probe_level.

Đệm 80 × 11 đi thẳng vào `belief.ts`. Phép chuẩn hoá nằm sẵn trong checkpoint.

Ngưỡng lấy từ `chosen_on_val` của `results/q2_decide_v3_v2_small_ens5.json`:
- D-thr v3 dừng sớm, một mạng: τ = 0,56, s = 0,6;
- B4b: e_thr = 0,657.

Đường dẫn checkpoint và JSON đều là tham số (`--q2-belief`, `--q2-decide-json`), nên đổi sang Q2 v4 chỉ cần thay checkpoint. ABORT có kiểm soát rời tấm theo kiểu push-off. Mất contact khi motor còn quay thì là SELF_ABORT: nhả tấm, level-hold rồi chuyển sang position.

**HoldPad (sửa):** ở chế độ diện tích càng, lò xo pháp tuyến nay tác dụng lên điểm càng thấp nhất. Trước đây nó tác dụng lên tâm đế, nằm cao 3–6 cm so với mặt tấm khi càng lệch góc, nên lò xo luôn kéo về −F_hold. Nhận định "trước đây load cell đọc −F_hold" là ƯL: lô có load cell đầu tiên đã chạy với bản sửa này.

### 13.1 Tín hiệu Gazebo so với `contact_phase.py` (ĐO, lô `q2_probe_m1`)

Cấu hình: p2y_perc_s0, xe đứng yên, tấm 65°, lịch đầy đủ RELAX → PULL0.3 → PULL0.6 → PULL0.9 rồi tắt motor. 9 vật liệu, mỗi vật liệu 1 lượt. Sim chạy 128 env với μ, F_hold, tilt, TW cố định. So bằng `ros/q2_signal_check.py`. Cột load cell lấy trung bình 50 ms cuối RELAX hoặc trung bình lúc giữ PULL.

| Tín hiệu | Gazebo/PX4 (ĐO) | `contact_phase.py` | Nhận xét |
|---|---|---|---|
| Load cell khi bám chắc (μ 0,9 hoặc 0,5 với F_hold 60,75 N; μ 0,9 với 24,3 N), RELAX / PULL0.3 / 0.6 / 0.9 | 0,1–2,3 / −1,7…−4,5 / −7,0…−10,1 / −12,9…−15,7 N | 5,6 / 2,9 / −2,5 / −7,8 N | Độ dốc theo PULL khớp (khoảng 5,6 N mỗi 0,3 g; lý thuyết m·0,3·g = 6,1 N). Gazebo lệch −4…−7 N, vì va chạm của càng với tấm đỡ một phần lực ép mà load cell (chỉ đo lực của HoldPad) không thấy (ƯL) |
| Load cell khi đang trượt (μ 0,2 / 0,5, hoặc F_hold 10 N) | 5,6–6,3 / 1,2–2,7 / −3,5…−4,4 / −9,7…−10,7 N | 5,6 / 2,9 / −2,4 / −7,8 N | Khớp trong khoảng 1–2 N |
| Năng lượng HP gia tốc tiếp tuyến (đặc trưng B4b) khi KHÔNG trượt | RELAX 0,004–0,011; PULL ≤ 0,0002 | RELAX 0,026; PULL 0,05–0,38 | Gazebo không có rung motor. Sim có rung motor (0,4 m/s² rms ở hover) |
| HP khi ĐANG trượt (HoldPad trượt 0,1–0,7 m/s) | 0,006–4,3 (RELAX), 0,004–1,3 (PULL) | 0,7–9,3 (RELAX), 1,2–10,5 (PULL) | Thấp hơn sim 3–100 lần. HoldPad cho trượt êm theo servo, không có rung trượt (sim giả định 2 m/s² rms mỗi m/s). Ở nhiều lượt trượt, giá trị nằm dưới e_thr = 0,657, nên B4b sẽ không phát hiện |
| Gyro rms ở RELAX | 0,10–0,32 rad/s | 0,70 rad/s | Sim cho body rate lúc chạm tắt dần (lấy mẫu tới 5 rad/s). Ở Gazebo, rate lúc chạm thấp hơn |
| Kết quả cuối lịch | μ 0,2 / F 24,3 và 10,1 N, μ 0,5 / F 10,1 N: SELF_ABORT (trượt khỏi mép hoặc tách). μ 0,5 / 24,3 N, μ 0,2 / 60,75 N, μ 0,9 / 10,1 N: trượt chậm 0,1–0,24 m/s rồi RƠI sau OFF | Cùng ba vật liệu: SELF_ABORT 69 / 87 / 107 trên 128 env. Ba vật liệu chắc: 128/128 còn bám | Phân loại vật liệu giống nhau |

Kết luận: trượt trong Gazebo chính là mô hình servo của HoldPad, không phải tín hiệu đo thật. IMU trong SITL sạch hơn sim nhiều (không có rung motor, không có rung trượt), nên cửa sổ C5 của SITL nằm ngoài phân phối của mạng niềm tin.

### 13.2 Mạng niềm tin trên cửa sổ SITL (ĐO, `q2_belief_probe.py`, `q2_channel_swap.py`)

Trên cửa sổ cuối RELAX của các lượt bám chắc (μ 0,9 / F 60,75 N và μ 0,5 / F 60,75 N, không trượt theo truth), `belief.ts` (v2_small seed 0) cho **p_hold ≈ 0,00 và p_slip ≈ 1,00**. Vì vậy D-thr dừng sớm sẽ ABORT ngay ở RELAX.

Trên cửa sổ cuối PULL0.3 / PULL0.6 của cùng các lượt, mạng cho p_hold 0,94–0,99 và p_slip ≤ 0,01, tức là đúng.

Hoán đổi từng nhóm kênh giữa cửa sổ SITL và cửa sổ sim (cùng vật liệu, cuối RELAX):

| Nhóm kênh | SITL lấy kênh của sim | Sim lấy kênh của SITL |
|---|---|---|
| load cell | (p_hold, p_slip) = (0,73; 0,15) cho lượt 1 | (0,03; 0,99) |
| IMU | không đổi | (0,71–0,72; 0,64–0,96) |
| Các nhóm còn lại (gyro, contact, thrust, phase, mức thử) | không đổi | không đổi |

Load cell là nguyên nhân chính: cuối RELAX, Gazebo đọc 0,1–2,3 N trong khi sim cho 5,6 N (mục 13.1). Nguồn thứ hai là gia tốc lúc chạm.

Hai cách sửa tín hiệu khác không có tác dụng ở RELAX:
- thay phần trước chạm bằng mô hình bay tự do của sim;
- thêm rung motor.

Hướng xử lý:
- HoldPad: cho load cell đo **tổng** lực pháp tuyến qua đế, gồm lực của HoldPad và lực va chạm của càng (lấy từ cảm biến contact), thay vì chỉ lực servo của HoldPad.
- Hoặc train lại mạng niềm tin với ngẫu nhiên hoá độ lệch và tỉ lệ load cell, IMU sạch (không rung), và phần trước chạm lấy từ rollout Q1.

## 14. Sửa theo review (2026-11-20)

| Mục | Sửa |
|---|---|
| D01 | HoldPad `touch_truth` tính v_n/v_t tại **điểm đế**, giống sim GPU: v_CoG + ω × r_pad, trừ vận tốc điểm mặt tấm ngay dưới đế, dùng vận tốc của bước trước va chạm. Vẫn ghi kèm v_n/v_t tại CoG và điểm đỡ của càng (12 trường). `rescore_pad.py` chấm lại offline các lô đã chạy: vận tốc đế dựng lại từ pose Gazebo (sai phân trung tâm, hoặc lùi nếu log dừng ngay lúc chạm). Kết quả ghi vào `results/sitl_*_padscored.json` |
| D02 | `Footprint.hh`: lấy điểm thấp nhất của hình chữ nhật càng **nằm trên tấm**. Góc treo ra ngoài mép không tính. Điểm đó sâu hơn 2 cm sau mặt tấm thì không hợp lệ. Chỉ gắn khi pad đang tiến vào hoặc gần đứng yên (đi ra không quá 5 cm/s). Có test C++ `test/footprint_test.cc`, gồm cả ca 25° trong review |
| D04 | `success` cần touch hợp lệ (trong đường bao) và không crash. Grace va chạm tấm tính theo stamp sự kiện, và chờ thêm 0,1 s để touch đến muộn vẫn được ghép. Lưu tên collision của va chạm tấm đầu tiên. Có 2 test mới |
| D08 | JSON có `provenance`: sha256 của `libHoldPad.so`, file `px4`, `policy.ts` và SDF sinh ra; git PX4; footprint thực dùng; biến môi trường; tham số PX4 đọc lại (MC_AIRMODE…). Launcher từ chối `.so` cũ hơn mã nguồn, hoặc `.so` thiếu footprint khi footprint được bật. Không đặt được `MC_AIRMODE` thì node thoát và lô dừng |
| D10 | Bỏ mẫu odom không hữu hạn (cả v và ω) hoặc có quaternion suy biến. Tiếp xúc hình học dự phòng xét hai phía (−5 cm ≤ d ≤ 2 cm) |
| D12 | Q2 MOTORS_OFF khi không còn contact thì chuyển thành SELF_ABORT. Disarm cưỡng bức chỉ gửi khi pad còn báo contact. Ở REST, mất contact khi còn armed thì rời tấm (level-hold) |
| D14 | `session()` kiểm lỗi `sitl_up.sh` tường minh. `sitl_down.sh` đợi tối đa 15 s cho tiến trình tắt hẳn rồi mới xoá file PID |

**Chấm lại offline lô chính tại điểm đế (ĐO, `rescore_pad.py`).** Với 0 m/s và 1,5 m/s, không lượt nào đổi kết quả. Ở 3 m/s có hai lượt đổi:

| Lô | Lượt | v_t tại CoG (HoldPad) | v_t tại đế |
|---|---|---|---|
| `main_v3` | #4 | 1,03 m/s | 0,37 m/s |
| `main_v3` | #7 | 1,59 m/s | 0,49 m/s |
| `main_p2y_prereview_v3` | #5 | 1,04 m/s | 0,46 m/s |
| `main_p2y_prereview_v3` | #9 | 1,01 m/s | 0,32 m/s |

Cả bốn lượt chuyển thành touch đạt. Chúng vẫn không tính là chu kỳ thành công, vì log dừng ngay lúc chạm nên không có REST để chấm.

v_t trung bình ở 3 m/s:
- `main_v3`: 0,66 m/s tại CoG, 0,19 m/s tại đế;
- p2y: 0,77 m/s tại CoG, 0,29 m/s tại đế.

Như vậy v_t cao ở 3 m/s chủ yếu là sai điểm đo.

Các lô Q2 và p2y chạy trước bản sửa được đổi tên thành `*_prereview`. Chúng đã chạy với HoldPad, eval và node trước các mục D01–D14 ở trên. Lô `q2_none` đầu tiên không hợp lệ từ lượt 11: EKF báo đang treo trong khi drone nằm dưới đất. Từ lúc đó evaluator kiểm thêm truth trước mỗi lượt.

## 15. Lô Q2 trong SITL sau khi sửa theo review (ĐÃ CHẠY)

Cấu hình chung:
- policy p2y_perc_s0, xe đứng yên, tấm 65°;
- lưới μ {0,2; 0,5; 0,9} × F_hold {10,13; 24,30; 60,75} N, tức 0,5 / 1,2 / 3 mg;
- 27 lượt mỗi luật (3 lượt mỗi vật liệu), `--eval-seed 2027`, REST 10 s;
- mã sau commit 50f6d26.

Có khởi động lại SITL 4–6 lần mỗi lô, sau các lượt rơi. Bảng tính bằng `ros/q2_report.py` và lưu ở `results/sitl_q2_report.json`. False-safe = số lượt rơi sau OFF / số lượt OFF.

| Luật | Đậu hợp lệ | False-safe (rơi / OFF) | ABORT có kiểm soát | SELF_ABORT | Crash khác | Về điểm chờ sau ABORT |
|---|---|---|---|---|---|---|
| dthr (v3, dừng sớm, τ 0,56, s 0,6) | 1/27 = 0,04 [0,01; 0,18] | 1/2 | 23/27 = 0,85 [0,68; 0,94] | 2/27 | 0/27 | 25/25 |
| dthr_min (phải qua PULL0.3, τ 0,54) | 8/27 = 0,30 [0,16; 0,49] | 6/14 = 0,43 [0,21; 0,67] | 1/27 | 12/27 | 0/27 | 13/13 |
| fsm_b4b (e_thr 0,657) | 7/27 = 0,26 [0,13; 0,45] | 6/13 = 0,46 [0,23; 0,71] | 9/27 | 4/27 | 1/27 | 13/13 |
| none (tắt motor ngay khi chạm) | 9/27 = 0,33 [0,19; 0,52] | 7/16 = 0,44 [0,23; 0,67] | 0/27 | 9/27 | 2/27 | 9/9 |
| *Sim tham chiếu, cùng lưới (`q2_sim_grid.py`, 128 ep/vật liệu)* | dthr / dthr_min / B4b: 0,33; none: 0,32 | dthr 0,008, dthr_min 0,005, B4b 0, none 0,67 | 0,63 / 0,21 / 0,63 | 0,03 / 0,46 / 0,03 | — | — |

Cách đọc (ĐO trừ chỗ ghi ƯL):
- Phân loại vật liệu khớp với sim. Ba vật liệu bám chắc (μ 0,5 / F 60,75; μ 0,9 / F 24,3; μ 0,9 / F 60,75) giữ được ở cả sim lẫn Gazebo. Sáu vật liệu còn lại trượt hoặc tách. Lý do: HoldPad dùng cùng công thức C2.
- **D-thr ABORT gần như mọi lượt ở RELAX**, kể cả trên vật liệu chắc. Mạng niềm tin nằm ngoài phân phối ở cửa sổ RELAX (mục 13.2): load cell khi bám chắc lệch so với sim và thay đổi theo hình dạng lúc chạm, gia tốc lúc chạm cũng khác.
- **dthr_min và B4b không thấy trượt chậm**: μ 0,5 / F 24,3 và μ 0,2 / F 60,75 trượt 0,1–0,2 m/s rồi rơi sau OFF, 3/3 lượt mỗi vật liệu ở cả hai luật. Trong Gazebo, trượt của HoldPad không sinh rung (mục 13.1), còn mạng và B4b đều dựa vào rung trượt (review Q2-03).
- **Baseline none không rơi 67% như sim mà 44%**, phần còn lại thành SELF_ABORT. Theo D12, khi mất contact trước lúc PX4 disarm xong, node không disarm mà rời tấm. Sim giả định OFF tức thì.
- **Kết luận:** chưa luật nào giảm được false-safe so với none trong SITL. Mọi khác biệt về false-safe giữa các luật đều nằm trong khoảng Wilson.

## 16. Lô chính p2y_perc_s0 sau khi sửa theo review (ĐÃ CHẠY)

Mã sau commit 50f6d26:
- touch chấm tại điểm đế;
- footprint chỉ tính phần càng nằm trên tấm;
- success phải có touch hợp lệ;
- JSON có provenance.

Mỗi tốc độ 10 lượt, rest 5 s, `--eval-seed 12345`. Mỗi lô 10/10, không lượt nào phải khởi động lại SITL:

| Lô | Chạm đạt | Giữ khi tắt motor | False-safe | Arm lại | Trọn chu kỳ | v_n / v_t tại đế (trung bình) |
|---|---|---|---|---|---|---|
| `sitl_main_p2y_v0` (0 m/s) | 10/10 [0,72; 1] | 10/10 | 0/10 | 10/10 | 10/10 [0,72; 1] | 1,17 / 0,13 m/s |
| `sitl_main_p2y_v15` (1,5 m/s) | 10/10 | 10/10 | 0/10 | 10/10 | 10/10 | 1,28 / 0,25 m/s |
| `sitl_main_p2y_v3` (3 m/s) | 10/10 | 10/10 | 0/10 | 10/10 | 10/10 [0,72; 1] | 1,22 / 0,29 m/s |

Ở 3 m/s:
- bản p2y trước review cho 7/10 chạm đạt và 6/10 trọn chu kỳ, chấm v_t tại CoG (trung bình 0,77 m/s);
- dựng lại tại đế thì v_t trung bình còn 0,29 m/s;
- bản p2m cho 8/10.

Hai bản p2y khác nhau cả mã chấm lẫn HoldPad (D02), nên phần cải thiện không quy riêng cho policy được.

## 17. Load cell của HoldPad = tổng lực pháp tuyến qua đế (2026-11-20)

`LoadCell.hh` cùng test `test/load_cell_test.cc` định nghĩa hai chế độ, chọn bằng tham số SDF `load_cell_mode` hoặc biến `PERCH_LOAD_CELL`:

| Chế độ | Giá trị đọc | Nhiễu | Bão hoà |
|---|---|---|---|
| `total` (mặc định) | Lực servo của HoldPad + thành phần pháp tuyến của lực va chạm Gazebo giữa drone và tấm. Lực va chạm lấy từ `ContactSensorData` của collision `plate_surface`, mỗi điểm chạm có wrench. Có dấu: + là ép | 3% FS, với FS = 2·m·g (`load_cell_fs_mg`) | ±FS |
| `servo` | Như trước 2026-11-20, chỉ lực servo | 3% giá trị đọc | không |

Topic `/perch/load_truth` phát `[t, servo_N, collision_N, số điểm có wrench]`. Evaluator ghi các giá trị này vào `truth_*.jsonl` (kind `load`).

Dấu của wrench được xác định bằng đo (ĐO). Lần chạy đầu cho thành phần va chạm khoảng −5 N lúc càng nằm yên trên tấm. Vì vậy wrench "của tấm" chính là lực đẩy lên drone, và đã sửa dấu theo đó (commit 30a0645).

**Lô `q2_probe_m3` (ĐO, so bằng `q2_signal_check.py`).** Với vật liệu bám chắc, độ lệch −3…−6 N không còn. Load cell ở RELAX / PULL0.3 / PULL0.6 / PULL0.9:

| Lượt | Gazebo (`total`) | Sim |
|---|---|---|
| μ 0,9 / F 60,75 | 7,3 / 3,1 / −2,7 / −8,0 N | 5,4 / 2,9 / −2,4 / −7,9 N |
| μ 0,9 / F 24,3 | 7,3 / 2,9 / −2,8 / −7,3 N | như trên |
| μ 0,5 / F 60,75 | 6,7 / 3,0 / −2,1 / −7,7 N | như trên |

Phần va chạm của càng là 2,9–4,5 N ở hai lượt μ 0,9. Lượt μ 0,5 / F 60,75 có phần va chạm ≈ 0 vì càng không tựa lên tấm. Đây chính là độ lệch đo được ở m1/m2.

Còn lệch: cuối RELAX, Gazebo cao hơn sim khoảng 1–2 N ở mọi vật liệu.

**Mạng niềm tin trên cửa sổ m3 (ĐO).**
- v2_small, cuối RELAX của các lượt bám chắc: p_hold 0,15 / 0,92 / 0,99 (m1 là khoảng 0).
- v4_small seed 0 trên mọi lượt bám chắc: p_hold 0,94–0,99 ở cả ba điểm quyết định.
- Nhưng v4 cũng cho p_hold ≥ 0,97 ở vài lượt đang trượt chậm (μ 0,2 / F 60,75), vì trượt chậm không có tín hiệu IMU.

### 17.1 Lô Q2 với load cell `total` (ĐÃ CHẠY)

Cấu hình:
- policy p2y_perc_s0, lưới vật liệu 3 × 3, `--eval-seed 2027`;
- D-thr dùng belief **v4_small seed 0**, ngưỡng lấy từ `q2_v4_small_main_lock.json` (`selection["D-thr"]["0.02"]`): τ 0,985, s 0,5, lịch v3;
- bảng từ `results/sitl_q2_lc_report.json`.

| Lô | n | Đậu hợp lệ | False-safe (rơi / OFF) | ABORT | SELF_ABORT |
|---|---:|---|---|---|---|
| `q2_dthr_lc` (0 m/s) | 27 | 5/27 = 0,19 [0,08; 0,37] | 3/8 = 0,38 [0,14; 0,69] | 10/27 | 8/27 |
| `q2_none_lc` (0 m/s) | 27 | 9/27 = 0,33 [0,19; 0,52] | 8/17 = 0,47 [0,26; 0,69] | 0 | 10/27 |
| `q2_dthr_lc_v15` (1,5 m/s) | 9 | 3/9 | 0/3 [0; 0,56] | 1/9 | 5/9 |
| `q2_none_lc_v15` (1,5 m/s) | 9 | 3/9 | 2/5 | 0 | 4/9 |

Lô `q2_dthr_lc` ghi `status: incomplete` chỉ vì lượt cuối (27) không về được điểm chờ. Cả 27 lượt đều đã được chấm.

So với `q2_dthr` (belief v2, load cell `servo`):
- D-thr với load cell đúng và v4 không còn ABORT hết ở RELAX: ABORT giảm từ 23/27 xuống 10/27, số lượt OFF tăng từ 2 lên 8. Phần này load cell giải thích được.
- False-safe vẫn 3/8: hai lượt μ 0,2 / F 60,75 trượt chậm và một lượt μ 0,9 / F 24,3 sát ngưỡng. Phần này load cell không giải thích được, vì trượt chậm không có tín hiệu IMU trong Gazebo.
- Với n hiện có, khác biệt về false-safe giữa D-thr và none nằm trong khoảng Wilson.

## Dashboard giám sát P7

Chạy từ thư mục gốc repo với venv `occwm`. **Replay** đọc log có sẵn:

```bash
PY=/home/sup/data/venvs/occwm/bin/python
LOG=/home/sup/data/perchpilot/sitl_eval
"$PY" perchpilot/ros/perch_dashboard.py --replay "$LOG/q2_b4e_qos" --trial 3
# Chạy lệnh sau ở terminal khác nếu muốn mở đồng thời hai lượt.
"$PY" perchpilot/ros/perch_dashboard.py --replay "$LOG/main_p2y_v15" --trial 1 --speed 1.0 --port 8766
```

Mở `http://127.0.0.1:8765/` (hoặc cổng đã chọn). Dùng thanh thời gian để dừng tại một mốc, **Phát** để tiếp tục và **Về đầu** để phát lại. Có thể mở trực tiếp `/?at=5.35` cho lượt ABORT hoặc `/?at=11.50` cho lượt chu kỳ trọn. `--speed` đổi tốc độ phát, trục biểu đồ vẫn theo thời gian của log.

**Live** đọc ROS 2 trên hệ thống đang chạy:

```bash
source /opt/ros/jazzy/setup.bash && source ~/ws_px4/install/setup.bash
export ROS_DOMAIN_ID=77
/home/sup/data/venvs/occwm/bin/python perchpilot/ros/perch_dashboard.py --live
```

Dashboard bind duy nhất `127.0.0.1` và truyền SSE tới trình duyệt. Live tham gia ROS domain 77 và lấy dữ liệu từ `/perchpilot/diag`, `/perch/hold_sensors` và `/px4_7/fmu/out/vehicle_odometry`. Dùng `perch_node` cùng checkout để có telemetry cửa sổ Q2. Khi có `eval_node` với publisher `/perchpilot/eval`, panel **truth (chỉ để chấm)** hiển thị đường bao lúc chạm và bộ đếm lô. Các trường bay và Q2 dùng telemetry của node điều khiển và cảm biến.

Replay đọc `node_ticks.jsonl`, `q2_samples.jsonl`, bản ghi diag do evaluator lưu và kết quả chấm trong `truth_ID.jsonl` / `trials.jsonl`. Khi có log console cùng tên lô trong thư mục replay hoặc `logs_agents/` cạnh `sitl_eval/`, dashboard dùng các mốc DEPART/DONE sau REST. Vị trí cũ có nhãn thời điểm lấy mẫu cuối. Ngưỡng B4e lấy từ bản ghi cấu hình `sitl_<tên-lô>.json` trong thư mục replay hoặc `perchpilot/results/`. Kênh thiếu mẫu hiện trống. Với lô có Q2, false-safe dùng chuẩn v1.2 tính từ lệnh OFF. Với lô placeholder, bộ đếm dùng kết quả sau disarm của evaluator.

```bash
/home/sup/data/venvs/occwm/bin/python -m pytest -q perchpilot/ros/test_perch_dashboard.py
```

## Node Rust

`rs/perch_core` chứa nhận thức, máy pha, Q2 và suy luận f32. `rs/perch_node` dùng `r2r` để nối cùng các topic PX4 và ROS 2. Node chạy policy tất định ở 100 Hz, lấy mẫu C5 ở 200 Hz và giữ schema `/perchpilot/diag`, `node_ticks.jsonl` và `q2_samples.jsonl`. Huấn luyện và xuất checkpoint dùng Python.

Build và xuất trọng số từ thư mục gốc repo.

```bash
source /opt/ros/jazzy/setup.bash
source ~/ws_px4/install/setup.bash
export LIBCLANG_PATH=/usr/lib/llvm-18/lib
cargo build --manifest-path rs/Cargo.toml --release -p perch_node --features ros

PY=/home/sup/data/venvs/occwm/bin/python
RUN=/home/sup/data/perchpilot/runs/p2y_perc_s0
BELIEF=/home/sup/data/perchpilot/runs/q2_belief_v7_full_s0
"$PY" ros/export_weights.py policy "$RUN/policy.ts"
"$PY" ros/export_weights.py belief "$BELIEF/belief.ts"
```

Bộ xuất nhận cả `best.pt` của hai mô hình. Mỗi bản xuất gồm `policy.rust.json` hoặc `belief.rust.json` và file `.bin` f32 little-endian. Header ghi shape, offset theo byte, metadata C1 hoặc C5 và SHA-256 của checkpoint nguồn. Khi dùng `--out DIR`, trỏ cờ `--run` hoặc `--q2-belief` của binary vào `DIR`. Launcher dùng đường dẫn C1 của evaluator, nên để bản xuất policy trong cùng thư mục với `policy.ts` khi chạy lô.

Chạy thử node qua ROS 2 với cảnh tổng hợp.

```bash
ROS_DOMAIN_ID=94 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  rs/target/release/perch_node --policy-run "$RUN" --dry-run --duration 2
```

Người vận hành SITL có thể chọn `PERCH_NODE_IMPL=rust` khi gọi `scripts/run_sitl_eval.sh`. Launcher mặc định dùng Python. Dùng `PERCH_NODE_ARGS` để truyền các cờ bổ sung như `--verify-rule dthr --q2-belief $BELIEF --q2-tau 0.44 --q2-s 1.01 --q2-dmin 0`. Luật `fsm_b4e` cần ba ngưỡng qua `--q2-slip-off` theo thứ tự RELAX, PULL0.3, PULL0.6. Với D-thr dùng belief 13 kênh hoặc B4e, để `PERCH_SLIP_SENSOR=1`.

Kiểm chứng ngoại tuyến bằng checkpoint và log hiện có.

```bash
CHECK=/tmp/perchpilot-rust-check
"$PY" ros/dump_reference.py --policy "$RUN" --belief "$BELIEF" --out "$CHECK"
cargo test --manifest-path rs/Cargo.toml --release
PERCH_RUST_CHECK="$CHECK" cargo test --manifest-path rs/Cargo.toml \
  --release -- --include-ignored --nocapture
cargo run --manifest-path rs/Cargo.toml --release -p perch_core --bin perch_bench -- "$CHECK"
```

Các lệnh kiểm chứng lõi chạy được khi chưa source ROS.

Kết quả đo (RTX 4060 Laptop, `p2y_perc_s0` + `q2_belief_v7_full_s0`):

| Kiểm chứng | Kết quả |
|---|---|
| Policy, Rust so với PyTorch | lệch action lớn nhất 5,0e-6 (đầu vào ngẫu nhiên), 2,6e-6 (obs SITL) |
| Belief net | lệch p_hold 8,3e-7, p_slip 1,5e-6 |
| Replay 7 kịch bản (chu kỳ trọn, B4e ABORT, D-thr OFF, mất tiếp xúc, …) | pha và quyết định khớp 100%, lệch lệnh ≤ 1,1e-5 |
| SITL 10 lượt, xe 1,5 m/s | chạm 10/10, trọn chu kỳ 9/10, false-safe 0/10 (lượt hỏng do harness hết giờ chờ EKF khi phục hồi) |
| Thời gian suy luận p95 trong SITL | 0,11 ms (Python 0,62 ms); jitter vòng 100 Hz p95 0,07 ms (Python 3,1 ms) |

