# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** 3.x / 2.x / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không 


### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`="cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về cả bức ảnh. Vì classification của checkpoint gán một lớp cho toàn bộ ảnh, không tạo một object riêng trong json. 
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint? Class đi kèm taxonomy/checkpoint, không phải do model tự nghĩ ra/
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp? `class_name` giúp người dễ đọc, `class_id` giúp máy dễ kiểm soát, `taxonomy_name` cho biết dùng lớp nào 
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp? Cần nói rõ chọn theo tiêu chí nào, để mọi annotator chọn thống nhất một lớp. 
6. Vì sao không được lấy `score` của model làm ground truth? vì `score` là độ đáng tin cậy của model trên prediction; nó không được người xác nhận và nó chỉ là prediction. 

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="person" ; `score`=0.912625 ; `bbox_xyxy`=[385.33,
      69.24,
      498.92,
      348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở [385.33, 69.24] , góc dưới-phải ở [348.22, 498.92], rộng 113.58 pixel,cao 279.68 pixel
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction? threshold = 0.20 có 17 vật thể, threshold = 0.60 có 6 vật thể
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải thực hiện các công việc kiểm tra và kiểm tra bounding của các box
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này). Mỗi đối tượng cần phát hiện phải được bao quanh bởi một bounding box nhỏ nhất có thể, ôm sát toàn bộ đối tượng, không cắt mất bất kỳ phần nào của đối tượng và hạn chế tối đa phần nền bên trong hộp. Mỗi đối tượng chỉ có một bounding box, kể cả khi các đối tượng chồng lấn lên nhau. Nếu đối tượng bị che khuất, chỉ vẽ box theo phần có thể quan sát được và không tự suy đoán phần bị che.


6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

Người ở dưới góc trái ảnh, bị cắt còn một bàn tay. mình sẽ hỏi lab coach chưa rõ một phần nhỏ của cơ thể như cánh tay/bàn tay có đủ điều kiện để gán nhãn person hay không. Cần xác nhận quy tắc trước khi áp dụng thống nhất cho toàn bộ dataset.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[446.0,70.0],[445.0,71.0],[444.0,71.0],[443.0,72.0],[442.0,72.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể? Polygon (đa giác) cung cấp thông tin chi tiết hơn về đường biên và hình dạng thực tế của vật thể so với bounding box. 
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id dùng để phân biệt từng cá thể vật thể riêng biệt trong cùng một class. nó k phải class id vì nó không xác định loại vật thể, cũng không phải tracking id vì nó không theo dõi một vật thể xuyên suốt nhiều frame
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo. 
Người gán nhãn phải vẽ polygon bám sát đường viền thực tế của vật thể, sử dụng đủ điểm để mô tả chính xác hình dạng, không bao gồm vùng nền xung quanh. Đối với vật thể bị che khuất, chỉ vẽ phần có thể quan sát được, không tự suy đoán phần bị che
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
ranh giới giữa chiếc bát và mặt bàn vì hai vật tiếp xúc trực tiếp, khiến việc xác định đường biên mask có thể gây nhầm lẫn. mình sẽ escalation lên lab coach để xác nhận cách xử lý vùng tiếp xúc, tránh mask chồng lấn và đảm bảo tính nhất quán trong dataset.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | một ảnh có nhiều nhãn  | khó quyết định nếu ảnh chứa nhiều loại đối tượng nhưng chỉ có một nhãn chính | xác định nội dung chính của ảnh và gán nhãn | kiểm tra xem gán có được gán đúng không |
| Phát hiện vật thể | mỗi đối tượng có một class id và một bounding box | Ảnh detection: person 0.61 ở mép trái chỉ hiển thị một phần cơ thể. Một số bị che khuất hoặc bị đè | Vẽ box bao quanh từng đối tượng, gán đúng class và xử lý vật bị che/cắt mép theo guideline | Kiểm tra class, vị trí box, box trùng lặp, false positives và false negatives. |
| Instance segmentation | Mỗi cá thể có class ID, instance ID và một mask/polygon thể hiện đường viền đối tượng | Ảnh segmentation: mask dining table 0.58 cần phân biệt rõ với các mask của bowl nằm trên mặt bàn; ranh giới giữa các vật tiếp xúc khá khó xác định | Vẽ mask bám sát đường viền từng vật thể, tách riêng các cá thể cùng class | Kiểm tra mask có đúng hình dạng, có chồng lấn sai, bỏ sót hoặc gộp nhầm các instance không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu) 

Tôi cam kết bảo mật toàn bộ dữ liệu của lab, không tự ý tải xuống, sao chép, chia sẻ hoặc tiết lộ dữ liệu cho người không có quyền truy cập.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Tôi sẽ dừng xử lý, không tự ý chỉnh sửa hoặc chia sẻ dữ liệu đó, đồng thời báo ngay cho Lab Coach hoặc người quản lý dự án để được hướng dẫn.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
