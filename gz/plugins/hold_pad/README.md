# HoldPad cho Gazebo Harmonic

`perchpilot::HoldPad` mô phỏng lực bám hữu hạn theo C2. Khi gắn, plugin giữ điểm đế bằng wrench tác dụng lên drone ở mỗi bước vật lý. Pose và vận tốc tiếp tục do bộ giải vật lý cập nhật.

## Build và nạp plugin

`sitl_up.sh` gọi `gz` từ `PATH`. Trên máy PerchPilot, shell đã source ROS Jazzy chọn Gazebo Sim 8.11.0 từ ROS vendor. Build và chạy thử trong cùng shell để CMake và server dùng cùng bộ thư viện. Chạy các lệnh sau từ gốc worktree.

```sh
p4=perchpilot/gz/plugins/hold_pad
which gz
gz sim --versions
cmake --fresh -S "$p4" -B "$p4/build" \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -Dgz-msgs10_DIR=/opt/ros/jazzy/opt/gz_msgs_vendor/lib/cmake/gz-msgs10
cmake --build "$p4/build"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$PWD/$p4/build${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"
```

Thư viện là `build/libHoldPad.so`. Export đường dẫn trong cùng shell dùng để khởi động Gazebo hoặc trước khi gọi launcher SITL. World `perch_world.sdf` đã khai báo plugin ở model `perch_cart`, với `drone_model=x500_7` và `drone_link=base_link`.

Nếu môi trường đã source ROS Jazzy và cần chọn bản Gazebo 8.15 từ gói hệ thống, dùng một shell có `PATH=/usr/bin:/bin`, bỏ `LD_LIBRARY_PATH` và `CMAKE_PREFIX_PATH`, đặt `GZ_CONFIG_PATH=/usr/share/gz` khi gọi CLI. Nếu CMake đã lưu đường dẫn ROS vendor, cấu hình lại với `--fresh` và `-Dgz-msgs10_DIR=/usr/lib/x86_64-linux-gnu/cmake/gz-msgs10` trên máy này.

## Hình học và tham số SDF

Model mang plugin cần link `plate` có collision hộp `plate_surface`. Mặt bám là mặt +Z của hộp trong hệ collision. Kích thước hộp xác định các mép. Tâm điểm đế dùng CoG của `drone_link` và dịch theo −Z thân. HoldPad tra world và drone từ `PreUpdate` và thử lại khi drone xuất hiện sau lúc khởi động.

| Tham số | Giá trị mặc định hoặc yêu cầu |
| --- | --- |
| `drone_model`, `drone_link` | Bắt buộc |
| `mu`, `f_hold_n` | Bắt buộc. Hệ số ma sát và lực hút tối đa (N) |
| `pad_offset` | 0,05 m. World x500 dùng 0,227 m theo đáy thanh càng của model SDF |
| `pad_radius` | 0,05 m. Cánh tay đòn để giới hạn mô-men giữ góc. Điều kiện nằm trong tấm xét tâm đế |
| `engage_vn_max` | 1,5 m/s. Vận tốc tiếp cận theo pháp tuyến, tính tương đối tại đế |
| `slip_gain` | 0,05 (m/s)/N |
| `sensor_noise_frac` | 0,03 |
| `sensor_delay_ms` | 10 ms, làm tròn lên bội số 5 ms |

## Mô hình lực

- `PreUpdate` chạy sau các system ưu tiên mặc định. Nó cộng lực trong `ExternalWorldWrenchCmd` trên các link thuộc model drone, kể cả rotor, trước khi ghi lực bám. Thrust là tổng lực này chia tổng khối lượng model. Khi thiếu lệnh lực, plugin ước lượng thrust từ hiệu vận tốc CoG qua một bước vật lý, trừ gia tốc trọng trường của world và lực bám bước đó chia khối lượng. Ước lượng có trễ một bước và có thể lẫn lực va chạm của bộ giải.
- Gia tốc tấm là sai phân vận tốc link `plate` trong hệ world. Công thức margin dùng `g=9,81 m/s²` như Python. Giá trị margin công bố được nhân với tổng khối lượng drone. Bước tính lực bù dùng trọng trường thực tế của world.
- Nếu đủ bám, bộ điều khiển lò xo–giảm chấn giữ tọa độ tiếp tuyến và góc tương đối lúc gắn. Lực pháp tuyến kéo về mặt tấm và bị chặn ở `−f_hold_n`. Margin tiếp tuyến quyết định hai chế độ giữ và trượt. Hiệu chỉnh quá độ để hãm drone là lực servo. Lúc trượt, servo tiếp tuyến bám vận tốc mục tiêu `slip_gain · |margin_N|` theo hướng thiếu lực. Tấm là mốc dẫn động.
- Khi thiếu lực hút hoặc tâm đế đi qua mép tấm, plugin nhả. Trạng thái ban đầu và sau reset cần lệnh cho phép gắn.

| Topic | Kiểu | Dữ liệu |
| --- | --- | --- |
| `/perch/hold_cmd` | `gz.msgs.Double` | 1 cho phép gắn, 0 nhả |
| `/perch/hold_params` | `gz.msgs.Double_V` | `[mu, f_hold_n]`, hữu hạn và không âm |
| `/perch/hold_params_applied` | `gz.msgs.Double_V` | Cặp giá trị đã áp dụng ở bước vật lý kế tiếp |
| `/perch/hold_truth` | `gz.msgs.Double_V` | `[t_sim_s, engaged, margin_N, slip_speed_mps]`. Tốc độ là độ lớn vận tốc trượt đo được khi đang trượt |
| `/perch/touch_truth` | `gz.msgs.Double_V` | `[t_sim_s, pad_u_m, pad_w_m, pad_n_m, align_deg, vn_mps, vt_mps, omega_radps]` khi đế đi vào vùng chạm. Vận tốc CoG tương đối và tốc độ góc lấy trước bước vật lý có va chạm |
| `/perch/hold_sensors` | `gz.msgs.Double_V` | `[t_sim_s, contact, f_normal_meas_N]`. Timestamp là lúc lấy mẫu |

Hai topic đầu ra lấy mẫu mỗi 5 ms thời gian mô phỏng ở các world đi kèm. `contact` báo đế ở vùng tiếp xúc hoặc đang gắn. Cảm biến lực dùng thành phần pháp tuyến có dấu của wrench HoldPad thực sự áp dụng, cộng nhiễu Gaussian có độ lệch chuẩn `sensor_noise_frac · |F|`. Mẫu được gửi sau khi đi qua bộ đệm vòng. Va chạm thụ động của bộ giải có thể xảy ra trước khi có lực HoldPad để đo.

```sh
gz topic -t /perch/hold_cmd -m gz.msgs.Double -p 'data: 1'
gz topic -t /perch/hold_cmd -m gz.msgs.Double -p 'data: 0'
gz topic -t /perch/hold_params -m gz.msgs.Double_V -p 'data: [0.2, 5.886]'
```

Cặp mới áp dụng đồng thời từ bước vật lý kế tiếp, kể cả khi đế đang gắn. Reset world khôi phục giá trị SDF.

## Kiểm thử

```sh
p4=perchpilot/gz/plugins/hold_pad
"$p4/build/hold_margin_test" > /tmp/hold_margin.csv
/home/sup/data/venvs/occwm/bin/python "$p4/test/compare_margin.py" /tmp/hold_margin.csv

p4_logs=$(mktemp -d /tmp/perchpilot-holdpad.XXXXXX)
python3 "$p4/test/run_headless.py" "$p4_logs"
```

Test headless kiểm tra RAM trước mỗi lượt, dùng `GZ_PARTITION=perchpilot_hold_pad_test` và khởi động `gz sim -s` với timeout 45 s. Box 2 kg rơi tự do tới tấm cố định 65°. Script dùng cùng SDF cho hai ca, gửi vật liệu qua `/perch/hold_params` và lệnh gắn trước khi chạy 6.500 bước. Ca mạnh dùng `mu=0,9`, `f_hold_n=58,86`. Ca yếu dùng `mu=0,2`, `f_hold_n=5,886`. Script chấm từ truth và pose, gồm bước dịch tại thời điểm gắn, độ trôi trong 5 s và diễn biến trượt rồi nhả ở mép tấm.
