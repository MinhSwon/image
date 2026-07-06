# BÁO CÁO DỰ ÁN CUỐI KỲ: NHẬP MÔN XỬ LÝ ẢNH SỐ
## CHỦ ĐỀ 2: IMAGE GRADIENT & HOG + SVM (PHÁT HIỆN NGƯỜI ĐI BỘ)

---

### TÓM TẮT DỰ ÁN
Báo cáo này trình bày kết quả thực hiện dự án cuối kỳ môn Nhập môn Xử lý ảnh số với đề tài **"Phát hiện người đi bộ trong video/webcam bằng phương pháp Image Gradient, Histogram of Oriented Gradients (HOG) và Support Vector Machine (SVM)"**. Hệ thống được phát triển sử dụng ngôn ngữ lập trình Python và thư viện xử lý ảnh OpenCV cùng các thư viện khoa học Scikit-Learn, Scikit-Image. Chúng tôi đã xây dựng chương trình chạy demo thời gian thực (realtime) ổn định bằng bộ phát hiện mặc định của OpenCV (pretrained HOG + SVM) đạt tốc độ cao và độ tin cậy tốt trên luồng video/webcam. Đồng thời, để nâng cao tính vận dụng và sáng tạo theo yêu cầu mở rộng, nhóm đã tự thu thập tập dữ liệu từ nguồn ngoài (INRIA Person Dataset qua Hugging Face với 300 ảnh positive chứa người và 300 ảnh negative nền không chứa người), thực hiện trích xuất vector đặc trưng HOG tùy chỉnh và huấn luyện thành công mô hình Linear SVM đạt độ chính xác (Accuracy) 96.67% trên tập kiểm thử (Test set) và 99.33% trên toàn bộ tập dữ liệu. Bản báo cáo cũng phân tích chi tiết cơ sở toán học và lý thuyết giải thuật của các kỹ thuật xử lý ảnh trung gian được hiển thị trực quan trong dự án.

---

### CHƯƠNG 1: GIỚI THIỆU

#### 1.1 Đặt vấn đề và tính cấp thiết của đề tài
Phát hiện người đi bộ (Pedestrian Detection) là một trong những bài toán kinh điển và quan trọng nhất trong lĩnh vực Thị giác máy tính (Computer Vision) và Xử lý ảnh (Image Processing). Bài toán này đóng vai trò nền tảng trong nhiều ứng dụng thực tiễn quan trọng, bao gồm:
*   **Hệ thống giám sát an ninh thông minh:** Tự động phát hiện và theo dõi sự xâm nhập trái phép của con người vào các khu vực cấm hoặc giám sát mật độ đám đông ở nơi công cộng.
*   **Hệ thống hỗ trợ lái xe tiên tiến (ADAS) và Xe tự hành:** Tự động nhận diện người đi bộ trên đường để đưa ra các cảnh báo phanh khẩn cấp hoặc điều khiển xe tránh va chạm, trực tiếp bảo vệ tính mạng con người.
*   **Phân tích hành vi khách hàng:** Trong các trung tâm thương mại hoặc cửa hàng bán lẻ, việc phát hiện và đếm số lượng người giúp đánh giá mật độ di chuyển và thói quen mua sắm.

Mặc dù các phương pháp Học sâu (Deep Learning) như YOLO, SSD hay Faster R-CNN đang chiếm ưu thế nhờ độ chính xác vượt trội, các phương pháp thị giác máy tính truyền thống (Traditional Computer Vision) kết hợp giữa trích xuất đặc trưng thủ công (Hand-crafted features) và học máy cổ điển (Classical Machine Learning) vẫn giữ nguyên giá trị khoa học to lớn. Trong đó, sự kết hợp giữa **Histogram of Oriented Gradients (HOG)** và **Support Vector Machine (SVM)** (được giới thiệu lần đầu bởi Navneet Dalal và Bill Triggs vào năm 2005) là một cột mốc lịch sử quan trọng. Kỹ thuật này nổi bật nhờ tính toán hiệu quả, cấu trúc toán học chặt chẽ, dễ giải thích và không yêu cầu năng lực tính toán cực lớn (như GPU) để huấn luyện hoặc chạy thử nghiệm.

#### 1.2 Mục tiêu đề tài
Dự án được xây dựng nhằm đạt được các mục tiêu sau:
1.  **Nghiên cứu lý thuyết:** Am hiểu sâu sắc bản chất toán học của toán tử vi phân tính toán Image Gradient, nguyên lý hoạt động của bộ mô tả đặc trưng HOG và cơ sở tối ưu hóa phân lớp của SVM.
2.  **Xây dựng chương trình Demo:** Phát triển một phần mềm hoàn chỉnh bằng Python và OpenCV có khả năng đọc luồng video từ file hoặc trực tiếp từ Webcam, thực hiện phát hiện người đi bộ theo thời gian thực và vẽ bounding box, hiển thị chỉ số FPS cũng như độ tin cậy.
3.  **Mở rộng & Sáng tạo:** Tự thiết kế quy trình tải dữ liệu ngoài từ Hugging Face, tự động tiền xử lý và cắt trích các mẫu ảnh, huấn luyện một mô hình Custom Linear SVM để phát hiện người đi bộ, thay vì chỉ sử dụng mô hình có sẵn của OpenCV.
4.  **Trực quan hóa:** Xuất các kết quả xử lý trung gian bao gồm ảnh Gradient Magnitude, Gradient Orientation và bản đồ HOG để phục vụ việc giải thích cơ chế vật lý của các thuật toán xử lý ảnh số.

---

### CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ GIẢI THUẬT

#### 2.1 Toán tử vi phân và Image Gradient (Độ dốc ảnh)
Gradient của một ảnh biểu thị sự thay đổi cường độ sáng theo các hướng trong không gian 2 chiều. Đối với một ảnh xám liên tục $I(x, y)$, gradient tại tọa độ $(x, y)$ là một vector hướng từ vùng tối sang vùng sáng và được định nghĩa bởi các đạo hàm riêng:

$$\nabla I(x, y) = \begin{bmatrix} G_x \\ G_y \end{bmatrix} = \begin{bmatrix} \frac{\partial I}{\partial x} \\ \frac{\partial I}{\partial y} \end{bmatrix}$$

Trong xử lý ảnh số (miền rời rạc), các đạo hàm riêng này được xấp xỉ bằng phép toán tích chập (convolution) ảnh với các mặt nạ vi phân (differential kernels). Trong dự án này, nhóm sử dụng **Toán tử Sobel** kích thước $3 \times 3$ để tính toán các thành phần gradient $G_x$ và $G_y$:
*   **Mặt nạ Sobel theo trục X ($S_x$)** (phát hiện biên dọc):
    $$S_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$$
    $$G_x(x, y) = I(x, y) * S_x$$
*   **Mặt nạ Sobel theo trục Y ($S_y$)** (phát hiện biên ngang):
    $$S_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$
    $$G_y(x, y) = I(x, y) * S_y$$

Từ hai thành phần này, ta tính được hai đại lượng vật lý quan trọng tại mỗi điểm ảnh:
1.  **Biên độ Gradient (Gradient Magnitude - $M(x,y)$):** Đại diện cho độ mạnh/độ nét của biên cạnh (sự thay đổi cường độ sáng đột ngột).
    $$M(x, y) = \sqrt{G_x(x, y)^2 + G_y(x, y)^2}$$
2.  **Hướng Gradient (Gradient Orientation - $\theta(x, y)$):** Góc của biên cạnh biểu thị hướng thay đổi nhanh nhất của cường độ sáng.
    $$\theta(x, y) = \arctan2(G_y(x, y), G_x(x, y))$$

Hướng $\theta$ được ánh xạ về khoảng $[0^\circ, 180^\circ]$ (unsigned gradient) trong bài toán trích xuất HOG để tránh ảnh hưởng của việc đổi màu sáng - tối ngược hướng.

#### 2.2 Đặc trưng Histogram of Oriented Gradients (HOG)
Mục đích của HOG là mô tả hình dáng và cấu trúc bề ngoài của đối tượng bằng sự phân bố của hướng gradient trong các vùng ảnh cục bộ. Quy trình trích xuất HOG bao gồm các bước chặt chẽ sau:

1.  **Tiền xử lý và Chuẩn hóa độ sáng ảnh:**
    *   Cửa sổ dò tìm chuẩn có kích thước $64 \times 128$ pixel (tỷ lệ $1:2$, tương ứng dáng người đứng thẳng).
    *   Áp dụng phép biến đổi căn bậc hai (Gamma Compression / Square-root normalization) trên ảnh để giảm bớt độ nhạy cảm của ảnh đối với các biến đổi ánh sáng môi trường hay bóng đổ:
        $$I_{new}(x, y) = \sqrt{I(x, y)}$$
2.  **Tính toán Gradient:**
    *   Sử dụng các toán tử lọc 1D đơn giản $[-1, 0, 1]$ và $[-1, 0, 1]^T$ để tính $G_x$ và $G_y$. Thực nghiệm cho thấy các bộ lọc vi phân nhỏ này cho hiệu quả trích xuất đặc trưng HOG tốt hơn các bộ lọc lớn (như Sobel) do chúng giữ lại được các thông tin biên chi tiết tốt hơn.
3.  **Phân chia ô (Cells) và Tạo Histogram:**
    *   Chia ảnh $64 \times 128$ thành các vùng nhỏ không chồng chập gọi là Cell có kích thước $8 \times 8$ pixel. Ảnh ban đầu sẽ chứa $(64/8) \times (128/8) = 8 \times 16 = 128$ cells.
    *   Tại mỗi cell, ta thống kê lược đồ hướng gradient (Histogram). Lược đồ này gồm 9 bins (thùng chứa) tương ứng với 9 khoảng góc từ $0^\circ$ đến $180^\circ$ (mỗi bin rộng $20^\circ$).
    *   Giá trị đóng góp (vote) của từng pixel vào bin hướng được trọng số hóa bởi biên độ gradient $M(x, y)$ tại chính pixel đó. Để tránh hiệu ứng răng cưa tại biên các ô và các bin hướng, phép nội suy tuyến tính (trilateral interpolation) theo cả tọa độ không gian $(x, y)$ và góc $\theta$ được áp dụng để phân bổ giá trị vote.
4.  **Gom nhóm khối (Blocks) và Chuẩn hóa độ sáng cục bộ (Block Normalization):**
    *   Các biến đổi ánh sáng diện rộng có thể làm thay đổi rất lớn biên độ gradient trung bình. Để khắc phục điều này, các cell lân cận được gom lại thành các khối lớn hơn gọi là Block. Project sử dụng kích thước block là $2 \times 2$ cells (tương đương $16 \times 16$ pixel). Các block này trượt chồng chập lên nhau với bước trượt (block stride) là 1 cell (8 pixel).
    *   Với kích thước ảnh $64 \times 128$, số block theo chiều ngang là $(8 - 2)/1 + 1 = 7$, số block theo chiều dọc là $(16 - 2)/1 + 1 = 15$. Tổng cộng ta có $7 \times 15 = 105$ blocks.
    *   Vector của mỗi block chứa $2 \times 2 = 4$ cells, mỗi cell có $9$ bins $\rightarrow$ mỗi block có $4 \times 9 = 36$ phần tử.
    *   Để chuẩn hóa, dự án sử dụng phương pháp **L2-Hys (L2-norm với Hysteresis)**:
        *   Bước 1: Tính chuẩn L2 thông thường:
            $$v_{norm} = \frac{v}{\sqrt{\|v\|_2^2 + \epsilon^2}}$$ (với $\epsilon$ là hằng số cực nhỏ tránh chia cho 0).
        *   Bước 2: Giới hạn (clip) các phần tử trong vector ở ngưỡng tối đa là 0.2 nhằm giảm ảnh hưởng của các biên độ gradient quá lớn cục bộ.
        *   Bước 3: Thực hiện chuẩn hóa L2 lại một lần nữa trên vector vừa bị clip.
5.  **Tạo Vector đặc trưng cuối cùng:**
    *   Ghép toàn bộ vector của 105 blocks lại với nhau để tạo thành vector đặc trưng HOG duy nhất cho cửa sổ ảnh $64 \times 128$:
        $$\text{Dung lượng vector} = 105 \text{ blocks} \times 36 \text{ giá trị/block} = 3780 \text{ chiều}$$

#### 2.3 Bộ phân loại Support Vector Machine (SVM)
Support Vector Machine là thuật toán học máy có giám sát tìm kiếm một siêu phẳng (hyperplane) phân tách tối ưu giữa hai lớp dữ liệu sao cho khoảng cách lề (margin) giữa các điểm dữ liệu gần nhất của hai lớp (gọi là các vector hỗ trợ - Support Vectors) đến siêu phẳng là lớn nhất.

Trong bài toán phát hiện người đi bộ sử dụng đặc trưng HOG, siêu phẳng phân hoạch là siêu phẳng tuyến tính (Linear SVM) nhằm đảm bảo tốc độ tính toán nhanh. Hàm quyết định phân lớp có dạng:

$$f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$$

Trong đó:
*   $\mathbf{x} \in \mathbb{R}^{3780}$ là vector đặc trưng HOG của cửa sổ dò tìm.
*   $\mathbf{w} \in \mathbb{R}^{3780}$ là vector trọng số tối ưu hóa cần học.
*   $b \in \mathbb{R}$ là sai số chệch (bias).

Nếu $f(\mathbf{x}) \ge 0$, mẫu ảnh được phân lớp là người đi bộ ($+1$, Positive); ngược lại, nếu $f(\mathbf{x}) < 0$, mẫu ảnh được phân lớp là nền ($0$ hoặc $-1$, Negative).
Khoảng cách từ điểm dữ liệu đến siêu phẳng quyết định độ tin cậy (confidence score) của phép dự đoán. Trong trường hợp cần tinh chỉnh ngưỡng phát hiện để giảm thiểu báo động giả (False Positives), ta có thể so sánh $f(\mathbf{x})$ với một ngưỡng $\tau$ (decision threshold) thay vì số 0.

---

### CHƯƠNG 3: THIẾT KẾ HỆ THỐNG VÀ ỨNG DỤNG DEMO

#### 3.1 Cấu trúc mã nguồn của dự án
Mã nguồn được tổ chức khoa học với các module thực hiện các nhiệm vụ chuyên biệt:
1.  `config.py`: Định nghĩa các tham số cấu hình chung như kích thước ảnh đầu vào, tham số HOG (9 hướng, ô 8x8, khối 2x2), đường dẫn các thư mục và mô hình lưu trữ.
2.  `features.py`: Cung cấp hàm tiền xử lý ảnh (chuyển grayscale, cân bằng lược đồ xám bằng `cv2.equalizeHist`) và trích xuất vector đặc trưng HOG từ ảnh thô hoặc đường dẫn ảnh.
3.  `detector.py`: Định nghĩa lớp `HOGSVMPedestrianDetector` sử dụng mô hình HOG + SVM có sẵn của OpenCV (`getDefaultPeopleDetector`). Đồng thời chứa các hàm hậu xử lý quan trọng:
    *   `non_max_suppression_with_scores`: Giải thuật NMS dựa trên diện tích chồng phủ IoU để loại bỏ các bounding box bị trùng lặp trên cùng một người.
    *   `remove_inner_boxes`: Giải thuật lọc bổ sung nhằm loại bỏ các box nhỏ nằm hoàn toàn trong một box lớn hơn, giúp khoanh vùng chính xác đối tượng.
    *   `draw_detections`: Vẽ bounding box, nhãn độ tin cậy và FPS lên màn hình hiển thị.
4.  `custom_detector.py`: Định nghĩa lớp `CustomHOGSVMSlidingWindowDetector` sử dụng kỹ thuật quét cửa sổ trượt (Sliding Window) đa tỷ lệ kết hợp với mô hình SVM tự huấn luyện (`models/custom_hog_svm.pkl`).
5.  `download_inria_hf_subset.py`: Tự động tải subset của bộ dữ liệu INRIA Person Dataset trên Hugging Face để phục vụ huấn luyện, lưu các ảnh dương (positive - chứa người) và cắt ngẫu nhiên các ảnh âm (negative - không chứa người) đưa về thư mục `dataset/`.
6.  `train_custom_svm.py`: Đọc tập dữ liệu mẫu, trích xuất đặc trưng HOG, phân chia tập train/test theo tỷ lệ 80/20 phân tầng (stratified split), chuẩn hóa dữ liệu bằng `StandardScaler` và huấn luyện mô hình Linear SVM qua thư viện Scikit-Learn.
7.  `evaluate_custom_svm.py`: Đánh giá hiệu năng mô hình trên toàn bộ tập dữ liệu.
8.  `visualize_gradient_hog.py`: Đọc một ảnh kiểm thử, tính toán và xuất các ảnh trung gian Sobel Magnitude, Orientation và HOG visualization để minh họa báo cáo.
9.  `run_realtime.py`: Chương trình điều khiển chính kết nối Webcam hoặc đọc Video, gọi các lớp detector để dự đoán và hiển thị giao diện đồ họa.

#### 3.2 Luồng xử lý dữ liệu và Thuật toán dò tìm đa tỷ lệ (Multi-scale Sliding Window)
Đối với mô hình tự huấn luyện (Custom SVM), vì nó chỉ nhận đầu vào cố định là ảnh $64 \times 128$, nhóm đã thiết lập giải thuật quét ảnh đa tỷ lệ để phát hiện người đi bộ với kích thước bất kỳ trong khung hình:

1.  **Tiền xử lý:** Nhận frame từ video/webcam, resize ảnh về chiều rộng tối đa (ví dụ 360 pixel) để tăng tốc độ quét và giảm gánh nặng tính toán cho CPU.
2.  **Khởi tạo danh sách các tỷ lệ ảnh (Scales):** Tạo các hệ số thu phóng ảnh (ví dụ: `1.0, 0.75, 0.55, 0.40`). Với mỗi hệ số, ảnh được resize tương ứng.
3.  **Quét cửa sổ trượt (Sliding Window):** Trên mỗi ảnh ở mỗi scale, sử dụng một cửa sổ quét kích thước cố định $64 \times 128$. Cửa sổ này dịch chuyển theo chiều ngang và dọc với bước nhảy `window_step` (ví dụ: 48 pixel).
4.  **Trích xuất đặc trưng và dự đoán:**
    *   Với mỗi cửa sổ trượt, trích xuất vector đặc trưng HOG (3780 chiều).
    *   Chuyển vector này qua Pipeline học máy (gồm chuẩn hóa `StandardScaler` và mô hình `LinearSVC`).
    *   Lấy điểm phân lớp từ hàm quyết định `decision_function`. Nếu điểm này lớn hơn ngưỡng quyết định `decision_threshold` (ví dụ: 0.8), lưu lại tọa độ cửa sổ (được quy đổi ngược lại kích thước ảnh gốc) cùng điểm tin cậy tương ứng.
5.  **Hậu xử lý:**
    *   **Non-Maximum Suppression (NMS):** Các cửa sổ phát hiện trùng lặp trên cùng một người được gom cụm. Nếu diện tích chồng chập IoU giữa hai box lớn hơn 0.35, giữ lại box có điểm tin cậy cao nhất và loại bỏ box còn lại.
    *   **Lọc hộp con (Remove Inner Boxes):** Nếu một box nhỏ bị chứa phần lớn bên trong một box lớn khác (tỷ lệ chứa > 70%), loại bỏ box nhỏ để tránh các bounding box phân mảnh.
6.  **Hiển thị:** Vẽ bounding box và nhãn lên khung hình gốc.

---

### CHƯƠNG 4: THÍ NGHIỆM VÀ HUẤN LUYỆN MÔ HÌNH CUSTOM

#### 4.1 Thu thập dữ liệu ngoài từ INRIA Person Dataset
Theo tiêu chí mở rộng và quy định sử dụng dữ liệu ngoài, dự án không lấy ảnh từ video demo để huấn luyện để tránh hiện tượng quá khớp (overfitting). Thay vào đó, nhóm đã viết kịch bản `download_inria_hf_subset.py` kết nối trực tiếp với mirror của **INRIA Person Dataset trên Hugging Face**.
*   **Ảnh Dương (Positive):** Tải 300 ảnh chứa người đi bộ, tự động cắt (crop) và đưa về kích thước chuẩn $64 \times 128$ pixel.
*   **Ảnh Âm (Negative):** Tải 300 ảnh phong cảnh, đường phố không chứa người đi bộ. Do ảnh nền ban đầu có kích thước lớn, kịch bản tự động cắt ngẫu nhiên (random crop) 2 vùng ảnh kích thước $64 \times 128$ trên mỗi bức ảnh nền để làm mẫu âm.
*   Kết quả thu được tập dữ liệu huấn luyện cân bằng gồm **600 mẫu** (300 Positive và 300 Negative).

#### 4.2 Cấu hình huấn luyện
Nhóm sử dụng thư viện `scikit-learn` để xây dựng luồng huấn luyện (Pipeline) tự động hóa:
*   **Chuẩn hóa dữ liệu:** Sử dụng `StandardScaler` để đưa các giá trị đặc trưng HOG về phân phối có trung bình bằng 0 và độ lệch chuẩn bằng 1. Việc này đặc biệt quan trọng đối với SVM giúp thuật toán hội tụ nhanh hơn và tránh việc các chiều đặc trưng có biên độ lớn lấn át các chiều đặc trưng khác.
*   **Mô hình:** Bộ phân loại `LinearSVC` với hằng số phạt sai số $C=1.0$, trọng số lớp tự động cân bằng (`class_weight='balanced'`), và số vòng lặp tối đa là 20,000 để đảm bảo hội tụ.
*   **Phân chia dữ liệu:** Chia ngẫu nhiên 80% cho tập huấn luyện (480 ảnh) và 20% cho tập kiểm thử (120 ảnh), phân bổ cân bằng nhãn giữa các tập.

#### 4.3 Mở rộng sáng tạo (20%): Mô hình nhận diện đối tượng mới (Face Detection)
Nhằm đạt tiêu chí "Năng lực vận dụng mở rộng & Sáng tạo", nhóm không chỉ dừng lại ở việc nhận diện người đi bộ mà còn tiến hành tự huấn luyện một **mô hình Linear SVM hoàn toàn mới để nhận diện khuôn mặt (Face Detection)**.
*   **Thu thập dữ liệu:** Nhóm sử dụng bộ dữ liệu `Labeled Faces in the Wild (LFW)` thông qua thư viện `scikit-learn` để lấy các ảnh mẫu dương (Positive - chứa khuôn mặt). Mẫu âm (Negative - không chứa khuôn mặt) được lấy ngẫu nhiên từ bộ ảnh nền của INRIA.
*   **Tiền xử lý:** Các ảnh được đưa về kích thước chuẩn của hệ thống là $64 \times 128$. Việc tái sử dụng kích thước cửa sổ của người đi bộ cho khuôn mặt là một thử nghiệm thú vị để kiểm chứng sự linh hoạt của hệ thống Sliding Window.
*   **Huấn luyện và Ứng dụng:** Mô hình được huấn luyện bằng luồng Pipeline tương tự như nhận diện người đi bộ và lưu tại `models/custom_hog_svm_face_model.pkl`. Nhóm cung cấp script `run_realtime_faces.py` cho phép người dùng chạy demo nhận diện khuôn mặt trực tiếp qua webcam với giao diện Bounding Box màu vàng đặc trưng.


---

### CHƯƠNG 5: KẾT QUẢ THỰC NGHIỆM VÀ THẢO LUẬN

#### 5.1 Các kết quả hình ảnh xử lý trung gian
Để kiểm chứng lý thuyết toán học tại Chương 2, nhóm đã chạy script `visualize_gradient_hog.py` trên ảnh mẫu. Các ảnh đầu ra trung gian được lưu trong thư mục `outputs/` bao gồm:
*   **Ảnh đầu vào resize (`outputs/input_resized.jpg`):** Ảnh gốc được chuyển về dạng xám và resize về kích thước chuẩn $64 \times 128$.
*   **Biên độ Gradient (`outputs/gradient_magnitude.jpg`):** Làm nổi rõ các đường biên, dáng người, viền quần áo. Các khu vực có cường độ sáng không đổi được chuyển về màu đen.
*   **Hướng Gradient (`outputs/gradient_orientation.jpg`):** Thể hiện các giá trị góc quay của gradient dưới dạng biểu đồ cường độ xám, cho thấy hướng biên thay đổi liên tục.
*   **Bản đồ trực quan HOG (`outputs/hog_visualization.jpg`):** Vẽ các lược đồ hướng dưới dạng các ngôi sao nhiều tia tại tâm của mỗi cell $8 \times 8$. Độ dài của tia biểu thị biên độ gradient của hướng tương ứng. Nhìn vào bản đồ này, mắt người vẫn có thể nhận ra cấu trúc hình dáng của đầu, vai, thân và chân người đi bộ.

#### 5.2 Đánh giá định lượng mô hình Custom SVM
Sau khi chạy huấn luyện, các chỉ số đánh giá được ghi nhận tại `outputs/custom_svm_report.txt`:

1.  **Kết quả trên Tập kiểm thử (120 mẫu):**
    *   **Độ chính xác (Accuracy):** **96.67%** (116/120 mẫu dự đoán đúng).
    *   **Ma trận nhầm lẫn (Confusion Matrix):**
        ```text
        [[59  1]   -> [Dự đoán đúng Background: 59, Sai Background thành Person: 1]
         [ 3 57]]  -> [Dự đoán sai Person thành Background: 3, Đúng Person: 57]
        ```
    *   **Độ chính xác chi tiết (Classification Report):**
        *   Lớp *background*: Precision = 0.95, Recall = 0.98, F1-score = 0.97
        *   Lớp *person*: Precision = 0.98, Recall = 0.95, F1-score = 0.97

2.  **Kết quả trên Toàn bộ tập dữ liệu (600 mẫu):**
    *   **Độ chính xác (Accuracy):** **99.33%** (596/600 mẫu đúng).
    *   **Ma trận nhầm lẫn (Confusion Matrix):**
        ```text
        [[299   1]
         [  3 297]]
        ```

Các con số thống kê này chứng minh mô hình tự huấn luyện có khả năng trích xuất đặc trưng HOG và phân loại cực tốt trên tập dữ liệu chuẩn của INRIA.

#### 5.3 So sánh mô hình OpenCV Pretrained và Custom Sliding-Window

| Đặc điểm | OpenCV Pretrained Detector | Custom Sliding-Window Detector |
| :--- | :--- | :--- |
| **Nguồn gốc SVM** | Huấn luyện trên tập dữ liệu INRIA gốc quy mô lớn của OpenCV | Huấn luyện trên subset 600 ảnh do nhóm tự tải |
| **Kỹ thuật quét ảnh** | Sử dụng thư viện tối ưu hóa bằng C++ của OpenCV (`detectMultiScale`) | Thuật toán quét cửa sổ trượt (Sliding Window) viết bằng Python |
| **Tốc độ khung hình (FPS)** | **Cao (15 - 28 FPS)**, chạy realtime rất mượt mà | **Thấp (~0.5 - 2 FPS)**, bị giật hình do vòng lặp quét thuần Python |
| **Độ chính xác thực tế** | Nhận diện tốt người đi bộ ở nhiều khoảng cách, ít bỏ sót | Nhận diện tốt ở cự ly gần và trung bình, nhạy cảm với các tỷ lệ quét |
| **Mục đích sử dụng** | Dùng để chạy demo chính thời gian thực trên webcam/video mẫu | Chứng minh và giải thích toàn bộ quy trình từ xử lý thô đến học máy |

---

### CHƯƠNG 6: NHẬN XÉT VÀ ĐÁNH GIÁ

#### 6.1 Ưu điểm của phương pháp
*   **Tính khoa học:** Phương pháp HOG + SVM thể hiện trực quan cách các thuật toán xử lý ảnh số sử dụng toán tử vi phân để chuyển đổi dữ liệu pixel thô sang dữ liệu đặc trưng hình dạng có ý nghĩa vật lý.
*   **Tính thực tiễn:** Chương trình demo chính sử dụng OpenCV pretrained chạy mượt mà trên máy tính cá nhân thông thường mà không cần phần cứng GPU hỗ trợ.
*   **Độ liêm chính học thuật:** Toàn bộ tập huấn luyện custom được tách biệt hoàn toàn khỏi video chạy demo thực tế. Nhóm tự tải và thiết lập bộ tham số thực nghiệm (kích thước bước nhảy window_step, ngưỡng ra quyết định decision_threshold, ngưỡng NMS).

#### 6.2 Hạn chế và nguyên nhân
*   **Tốc độ của Custom model:** Bộ phát hiện tự huấn luyện chạy khá chậm. Nguyên nhân là do phép quét sliding window đa tỷ lệ được thực thi qua các vòng lặp lồng nhau trong Python, đồng thời việc trích xuất đặc trưng HOG cho hàng trăm cửa sổ quét trên mỗi frame tạo ra gánh nặng tính toán lớn. Để khắc phục tạm thời, nhóm đã đưa ra giải pháp giảm kích thước tối đa của ảnh đầu vào (`--custom-max-width 360`) và tăng bước nhảy cửa sổ (`--custom-step 48`).
*   **Nhạy cảm với bối cảnh:** Cả mô hình OpenCV và mô hình Custom đều có thể phát sinh lỗi phát hiện nhầm (False Positive) khi nền ảnh chứa quá nhiều cạnh sắc thẳng đứng (như hàng rào, cột điện, thân cây) do chúng có đặc trưng phân bố HOG tương đối giống với cấu trúc dọc của cơ thể người.

---

### CHƯƠNG 7: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

#### 7.1 Kết luận
Dự án cuối kỳ môn Nhập môn Xử lý ảnh số - Chủ đề 2 đã hoàn thành xuất sắc toàn bộ các mục tiêu đặt ra:
*   Giải thích rõ ràng các cơ sở toán học và thuật toán (Image Gradient, Sobel, HOG, SVM).
*   Phát triển thành công chương trình demo chạy thời gian thực phát hiện người đi bộ trên video/webcam với bounding box, nhãn và FPS hiển thị trực quan.
*   Tự động hóa tải tập dữ liệu ngoài INRIA và huấn luyện thành công mô hình Custom SVM đạt độ chính xác cao (96.67%).
*   **Điểm sáng tạo:** Hoàn thành xuất sắc yêu cầu mở rộng bằng cách tự huấn luyện thêm một mô hình nhận diện đối tượng mới (Khuôn mặt - Face Detection) dựa trên tập dữ liệu LFW, tích hợp thành công vào pipeline hệ thống.
*   Xuất được các kết quả trung gian làm minh chứng lý thuyết.

#### 7.2 Hướng phát triển tương lai
*   **Tối ưu hóa mã nguồn:** Sử dụng thư viện Numba hoặc Cython để biên dịch mã quét sliding window của Python sang mã máy, giúp tăng tốc độ thực thi của mô hình Custom.
*   **Khai phá mẫu âm khó (Hard Negative Mining):** Thu thập các vùng ảnh nền bị nhận diện nhầm là người (False Positives), đưa chúng ngược trở lại tập dữ liệu huấn luyện để dạy mô hình SVM nhận diện tốt hơn, tránh lỗi nhiễu.
*   **Tích hợp mô hình học sâu:** Nghiên cứu tích hợp thêm bộ phát hiện dựa trên mạng nơ-ron tích chập (YOLO) để tiến hành so sánh đối chiếu hiệu năng giữa xử lý ảnh truyền thống và học sâu hiện đại.
