# x500_perch

Bản sao model `x500` từ `~/PX4-Autopilot/Tools/simulation/gz/models/x500`. Model dùng chung `model://x500_base` và các mesh có sẵn trong PX4. Launcher đưa cả thư mục model của PerchPilot và của PX4 vào `GZ_SIM_RESOURCE_PATH`.

## Tỉ lệ lực đẩy/trọng lượng

Dùng khối lượng danh nghĩa 2,064 kg, gia tốc trọng trường 9,81 m/s², bốn rotor và tốc độ tối đa 1000 rad/s. Hằng số gốc `C₀ = 8,54858 × 10⁻⁶` N/(rad/s)².

```text
TW_gốc = 4 C₀ ω_max² / (m g) = 1,68879
k = TW_mục_tiêu / TW_gốc = 2,26 m g / (4 C₀ ω_max²) = 1,338237415
motorConstant = k C₀ = 1,14400296 × 10⁻⁵ N/(rad/s)² cho cả 4 rotor
F_max = 4 · motorConstant · ω_max² = 45,7601 N
```

`momentConstant` vẫn là 0,016 m. Gazebo tính mô-men phản lực từng rotor theo `−turningDirection · momentConstant · F_rotor`. Vì vậy cùng tốc độ rotor thì mô-men phản lực từng rotor và chênh lệch mô-men do lệnh yaw cùng tăng ×1,338. Hai cặp quay ngược chiều triệt tiêu yaw khi tốc độ đều nhau. Tỉ lệ mô-men/lực đẩy giữ nguyên. [Mã MulticopterMotorModel của Gazebo](https://github.com/gazebosim/gz-sim/blob/gz-sim8_8.11.0/src/systems/multicopter_motor_model/MulticopterMotorModel.cc#L609-L617)

## Ánh xạ thrust

Ở trạng thái đã arm, với bốn rotor nhận cùng mức điều khiển `u ∈ [0,1]` và sau khi rotor ổn định, `ω = 150 + 850 u` rad/s và `F = 4 · motorConstant · ω²` N. `u = 0` ở đây là tốc độ idle 150 rad/s. Công thức áp dụng cho `THR_MDL_FAC = 0` và giới hạn ESC của airframe 4001.

| u | ω (rad/s) | Tổng lực đẩy (N) |
|---:|---:|---:|
| 0,00 | 150 | 1,0296 |
| 0,10 | 235 | 2,5271 |
| 0,20 | 320 | 4,6858 |
| 0,30 | 405 | 7,5058 |
| 0,40 | 490 | 10,9870 |
| 0,50 | 575 | 15,1294 |
| 0,60 | 660 | 19,9331 |
| **0,606106 (hover)** | **665,190** | **20,2478** |
| 0,70 | 745 | 25,3980 |
| 0,80 | 830 | 31,5241 |
| 0,90 | 915 | 38,3115 |
| 1,00 | 1000 | 45,7601 |

Điểm hover cân bằng lực đẩy với trọng lượng `mg`. Tính cùng công thức cho x500 gốc cho `u_hover = 0,728831`, khớp với ước lượng ≈0,73 trước đó.

```python
u = clip((sqrt(max(thrust_N, 0.0) / (4 * 1.14400296e-5)) - 150.0) / 850.0, 0.0, 1.0)
```

`perch_core.ActionScale.acc_to_u(thrust_N / mass)` dùng phép nghịch đảo này khi chọn `--thrust-model x500_perch_gz`. `action_to_px4` kẹp kết quả về [0,1] rồi gửi `thrust_body[2] = -u` theo FRD.

## PX4

- `MPC_THR_HOVER` có giá trị khởi tạo 0,60 trên airframe 4001. Hover vật lý ước lượng 0,606 ở tư thế ngang. Có thể dùng giá trị đó để khởi tạo khi đo SITL. Kiểm tra giá trị lưu từ phiên dùng x500 gốc và hội tụ của bộ ước lượng hover khi `MPC_USE_HTE` bật.
- `CA_ROTOR0_CT` đến `CA_ROTOR3_CT` là hệ số hiệu lực tương đối của bộ phân bổ PX4. Mã dùng `thrust = CT · axis` và `moment = CT · (position × axis − KM · axis)`. Với bốn rotor cùng tăng hệ số, phép chuẩn hoá ma trận trộn loại ảnh hưởng của hệ số chung. Giữ CT mặc định ban đầu. `CA_ROTOR*_KM` hiện là ±0,05 trong airframe 4001 và cũng giữ nguyên. Kiểm tra đáp ứng rate/yaw và bão hoà trong P5.
- `SIM_GZ_EC_MIN1..4=150`, `SIM_GZ_EC_MAX1..4=1000` và `THR_MDL_FAC=0` là điều kiện của bảng. Lệnh rate lớn có thể khiến từng rotor chạy ở các `u` khác nhau và chạm giới hạn.

Các thay đổi dành cho PX4 nằm trong [px4_patches](../../../px4_patches/).
