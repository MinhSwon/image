# KỊCH BẢN THUYẾT TRÌNH (DỰ KIẾN 15-20 PHÚT)
## CHỦ ĐỀ 2: IMAGE GRADIENT, HOG & SVM (PHÁT HIỆN NGƯỜI ĐI BỘ VÀ KHUÔN MẶT)

---

### Slide 1: Tiêu đề & Giới thiệu (1 phút)
**Nội dung hiển thị:**
* Tên đề tài: Phát hiện người đi bộ và đối tượng mở rộng (Khuôn mặt) bằng Image Gradient, HOG và SVM.
* Thành viên nhóm, Giáo viên hướng dẫn.

**Lời thoại thuyết trình:**
"Chào thầy và các bạn. Hôm nay, nhóm chúng em xin trình bày báo cáo cuối kỳ môn Nhập môn Xử lý ảnh số - Chủ đề 2. Đề tài của nhóm là phát triển một ứng dụng Thị giác máy tính truyền thống để phát hiện người đi bộ theo thời gian thực dựa trên 3 kỹ thuật cốt lõi: Image Gradient, đặc trưng HOG và bộ phân loại học máy SVM. Đặc biệt, nhóm cũng xin trình bày phần mở rộng sáng tạo: tự huấn luyện một mô hình phát hiện khuôn mặt."

---

### Slide 2: Đặt vấn đề & Mục tiêu (2 phút)
**Nội dung hiển thị:**
* Ứng dụng: Giám sát an ninh, xe tự hành (ADAS), phân tích hành vi.
* Mục tiêu 1: Hiểu và hiện thực hóa cơ sở toán học của xử lý ảnh.
* Mục tiêu 2: Demo Realtime (Webcam/Video).
* Mục tiêu 3 (Mở rộng): Tự train mô hình Custom SVM cho đối tượng mới (Khuôn mặt).

**Lời thoại thuyết trình:**
"Phát hiện đối tượng là nền tảng của nhiều công nghệ như xe tự hành hay camera an ninh. Mặc dù Deep Learning đang rất phổ biến, nhưng việc làm chủ kỹ thuật HOG + SVM là cực kỳ quan trọng vì nó giúp chúng ta hiểu rõ 'bản chất vật lý' của bức ảnh thay vì coi nó là một hộp đen. Mục tiêu của nhóm không chỉ là dùng tool có sẵn, mà là xây dựng quy trình từ A đến Z: từ xử lý điểm ảnh thô, rút trích đặc trưng, đến việc tự thu thập dữ liệu và huấn luyện mô hình."

---

### Slide 3: Tổng quan hệ thống (1 phút)
**Nội dung hiển thị:**
* Sơ đồ: Ảnh đầu vào $\rightarrow$ Resize & Grayscale $\rightarrow$ Tính Gradient $\rightarrow$ Trích xuất HOG $\rightarrow$ Đưa qua SVM $\rightarrow$ Hậu xử lý (NMS) $\rightarrow$ Xuất Bounding Box.

**Lời thoại thuyết trình:**
"Hệ thống hoạt động qua một luồng (pipeline) 5 bước. Ảnh hoặc video từ webcam sẽ được thu nhỏ và chuyển sang ảnh xám. Sau đó, đạo hàm không gian được tính để tìm Gradient. Từ Gradient, ta trích xuất HOG – linh hồn của thuật toán. Vector HOG này được đưa cho SVM chấm điểm, cuối cùng thuật toán Non-Maximum Suppression sẽ lọc các khung hình trùng lặp để cho ra kết quả cuối cùng."

---

### Slide 4: Cơ sở lý thuyết - Image Gradient (3 phút)
**Nội dung hiển thị:**
* Toán tử Sobel $3 \times 3$ (Trục X và Y).
* Biên độ $M(x,y)$ và Hướng $\theta(x,y)$.
* Chèn ảnh minh họa: `outputs/gradient_magnitude.jpg` và `outputs/gradient_orientation.jpg`.

**Lời thoại thuyết trình:**
"Bước đầu tiên để máy tính 'thấy' hình dáng là tìm biên cạnh. Nhóm sử dụng mặt nạ Sobel nhân tích chập với ảnh. Như thầy và các bạn thấy trên màn hình:
- Ảnh Magnitude (bên trái) làm nổi bật các đường viền quần áo, cơ thể (do sự thay đổi ánh sáng lớn).
- Ảnh Orientation (bên phải) biểu diễn góc của các biên đó. Gradient rất quan trọng vì nó không bị ảnh hưởng nhiều khi ta đổi màu áo hay thay đổi độ sáng chung của môi trường."

---

### Slide 5: Cơ sở lý thuyết - Đặc trưng HOG (4 phút)
**Nội dung hiển thị:**
* Cửa sổ: $64 \times 128$.
* Cell: $8 \times 8$ pixel. Block: $2 \times 2$ cells.
* Histogram: 9 bins (0-180 độ).
* Chuẩn hóa L2-Hys cục bộ.
* Chèn ảnh minh họa: `outputs/hog_visualization.jpg`.

**Lời thoại thuyết trình:**
"Thay vì dùng pixel thô, ta nhóm chúng lại thành các Cell $8 \times 8$ pixel. Tại mỗi cell, ta đếm xem có bao nhiêu pixel hướng theo góc $0^\circ, 20^\circ, 40^\circ \dots$ (tổng cộng 9 hướng). Cứ 4 cells ta gom thành 1 Block và chuẩn hóa L2-Hys để chống nhiễu sáng.
Kết quả của một ảnh $64 \times 128$ là một vector khổng lồ 3780 chiều. Nhìn vào ảnh minh họa HOG (các đường chéo hình ngôi sao), ta thấy rõ hình bóng của phần đầu, vai và thân người. Đây chính là cách máy tính 'hiểu' hình dáng."

---

### Slide 6: Cơ sở lý thuyết - Thuật toán phân lớp SVM (2 phút)
**Nội dung hiển thị:**
* SVM tìm siêu phẳng (Hyperplane) chia cắt không gian 3780 chiều.
* $f(x) = w^T x + b$.
* Tối ưu hóa Soft-margin (C=1.0) và cân bằng lớp (Balanced weights).

**Lời thoại thuyết trình:**
"Với vector HOG 3780 chiều, bài toán trở thành tìm một mặt phẳng trong không gian 3780 chiều để chia làm 2 phe: Người và Nền. Nhóm sử dụng Linear SVM vì tốc độ phân lớp của hạt nhân tuyến tính cực kỳ nhanh, rất phù hợp cho realtime. Chúng em cũng thiết lập thông số cân bằng lớp (class_weight='balanced') để mô hình không bị thiên lệch dù số lượng ảnh nền thường nhiều hơn ảnh người."

---

### Slide 7: Thuật toán Quét cửa sổ trượt - Sliding Window & NMS (3 phút)
**Nội dung hiển thị:**
* Khái niệm Kim tự tháp ảnh (Image Pyramid) & Quét cửa sổ.
* Hiện tượng Overlapping Boxes.
* Thuật toán Non-Maximum Suppression (NMS) với ngưỡng IoU.

**Lời thoại thuyết trình:**
"Do mô hình chỉ nhận kích thước chuẩn, nhóm triển khai kỹ thuật Kim tự tháp ảnh: thu nhỏ ảnh dần dần và dùng cửa sổ quét khắp ảnh (Sliding Window).
Tuy nhiên, quét như vậy sẽ khiến 1 người bị phát hiện nhiều lần ở các tọa độ hơi lệch nhau. Nhóm đã tự code thuật toán NMS: so sánh diện tích giao nhau (IoU) của các hộp. Hộp nào có độ tin cậy lớn nhất sẽ được giữ lại, các hộp trùng lặp xung quanh bị xóa đi, tạo ra một bounding box duy nhất và gọn gàng nhất."

---

### Slide 8: Ứng dụng Thực hành - Phần 1: OpenCV Pretrained (1 phút)
**Nội dung hiển thị:**
* Chạy realtime trên Webcam/Video bằng thư viện C++ của OpenCV.
* Tốc độ: Cao (15 - 30 FPS).
* Lệnh chạy: `python src/run_realtime.py --source 0`

**Lời thoại thuyết trình:**
"Về mặt triển khai, ứng dụng có thể chạy realtime qua webcam. Nếu dùng bộ Pretrained HOG+SVM của OpenCV, hệ thống đạt tốc độ rất mượt mà. Tuy nhiên, nếu chỉ dùng hàm có sẵn thì chưa phản ánh hết kỹ năng thực hành, do đó nhóm đã tiến tới phần 2."

---

### Slide 9: Ứng dụng Thực hành - Phần 2: Tự huấn luyện (Custom SVM) (2 phút)
**Nội dung hiển thị:**
* Code tải INRIA Dataset (Hugging Face): 300 Positive, 300 Negative.
* Kịch bản huấn luyện: StandardScaler + LinearSVC.
* Kết quả test set: Accuracy 96.67%.

**Lời thoại thuyết trình:**
"Đây là điểm nhấn của dự án. Nhóm viết code tự tải bộ INRIA Dataset chuẩn, tự cắt ảnh nền ngẫu nhiên. Trích xuất 600 vector HOG và tự build Pipeline huấn luyện.
Kết quả cho thấy mô hình tự train đạt độ chính xác lên tới 96.67% trên tập Test. Mô hình này được lưu lại thành file `.pkl` để sử dụng độc lập."

---

### Slide 10: SÁNG TẠO MỞ RỘNG (20%) - Nhận diện khuôn mặt (2 phút)
**Nội dung hiển thị:**
* Train mô hình phân loại đối tượng MỚI.
* Dataset: Labeled Faces in the Wild (LFW).
* Tái sử dụng HOG và SVM.
* Lệnh demo: `python src/run_realtime_faces.py --source 0`

**Lời thoại thuyết trình:**
"Để đáp ứng 20% điểm vận dụng mở rộng sáng tạo, thay vì chỉ nhận diện người đi bộ, nhóm đã chứng minh tính tổng quát của hệ thống bằng cách huấn luyện mô hình nhận diện Khuôn mặt (Face Detection). Nhóm sử dụng bộ dữ liệu khuôn mặt LFW, đưa qua đúng quy trình HOG+SVM đã xây dựng. Kết quả là hệ thống hoàn toàn có thể tìm ra bounding box khuôn mặt trực tiếp trên webcam. Điều này minh chứng hệ thống có thể mở rộng cho Biển báo, Xe cộ, hay bất cứ vật thể nào có hình dáng cố định."

---

### Slide 11: Tổng kết và Trực tiếp Demo (Vòng cuối)
**Nội dung hiển thị:**
* Ưu điểm: Hiểu sâu lý thuyết, chạy realtime ổn định, có mô hình tự train.
* Hạn chế: Thuật toán Sliding Window code bằng Python thuần còn chậm.
* Hướng phát triển: Kết hợp dò tìm thông minh hơn, đối chiếu với YOLO.
* (Mở chương trình lên chạy Demo trực tiếp cho Giáo viên xem).

**Lời thoại thuyết trình:**
"Nhìn chung, dự án đã bám sát 100% yêu cầu chuyên môn, giữ vững liêm chính học thuật qua việc tự code các logic lõi và hiểu rõ từng thông số. Dù thuật toán sliding window bằng Python chạy còn hơi chậm so với C++, nhưng nó mang lại giá trị sư phạm cực lớn.
Xin cảm ơn thầy và các bạn đã lắng nghe. Sau đây nhóm xin phép chạy trực tiếp phần mềm từ Webcam để mọi người cùng xem kết quả!"
