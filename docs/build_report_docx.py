from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "BaoCao_HOG_SVM.docx"


BLUE = RGBColor(46, 116, 181)
DARK_BLUE = RGBColor(31, 77, 120)
MUTED = RGBColor(85, 85, 85)
LIGHT_GRAY = "F2F4F7"
BLUE_GRAY = "E8EEF5"


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_width(cell, width_dxa):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.find(qn("w:tcMar"))
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for name, value in [("top", top), ("start", start), ("bottom", bottom), ("end", end)]:
        element = tc_mar.find(qn(f"w:{name}"))
        if element is None:
            element = OxmlElement(f"w:{name}")
            tc_mar.append(element)
        element.set(qn("w:w"), str(value))
        element.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_table_geometry(table, widths):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")

    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")

    tbl_grid = tbl.tblGrid
    if tbl_grid is None:
        tbl_grid = OxmlElement("w:tblGrid")
        tbl.append(tbl_grid)
    for child in list(tbl_grid):
        tbl_grid.remove(child)
    for width in widths:
        grid_col = OxmlElement("w:gridCol")
        grid_col.set(qn("w:w"), str(width))
        tbl_grid.append(grid_col)

    for row in table.rows:
        for index, width in enumerate(widths):
            cell = row.cells[index]
            set_cell_width(cell, width)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_page_number(paragraph):
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = "PAGE"
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr_text)
    run._r.append(fld_char2)


def configure_styles(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.font.size = Pt(11)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10

    for style_name, size, color, before, after in [
        ("Heading 1", 16, BLUE, 16, 8),
        ("Heading 2", 13, BLUE, 12, 6),
        ("Heading 3", 12, DARK_BLUE, 8, 4),
    ]:
        style = doc.styles[style_name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = color
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)


def add_title_page(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(70)
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run("BÁO CÁO CUỐI KỲ")
    run.bold = True
    run.font.size = Pt(16)
    run.font.color.rgb = MUTED

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(12)
    run = p.add_run("Phát Hiện Người Đi Bộ Bằng Image Gradient, HOG và SVM")
    run.bold = True
    run.font.size = Pt(24)
    run.font.color.rgb = RGBColor(0, 0, 0)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(26)
    run = p.add_run("Chủ đề 2: Image Gradient & HOG + SVM")
    run.font.size = Pt(13)
    run.font.color.rgb = MUTED

    rows = [
        ("Môn học", "Nhập môn Xử lý ảnh số"),
        ("Ngôn ngữ / thư viện", "Python, OpenCV, scikit-image, scikit-learn"),
        ("Demo chính", "Realtime pedestrian detection trên video/webcam"),
        ("Model mở rộng", "Custom Linear SVM train từ 80 positive và 120 negative"),
    ]
    table = doc.add_table(rows=0, cols=2)
    table.style = "Table Grid"
    set_table_geometry(table, [2300, 7060])
    for label, value in rows:
        cells = table.add_row().cells
        cells[0].text = label
        cells[1].text = value
        set_cell_shading(cells[0], LIGHT_GRAY)
        for cell in cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
                for run in paragraph.runs:
                    run.font.size = Pt(10.5)

    doc.add_page_break()


def add_header_footer(doc):
    section = doc.sections[0]
    header = section.header.paragraphs[0]
    header.text = "HOG + SVM Pedestrian Detection"
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in header.runs:
        run.font.size = Pt(9)
        run.font.color.rgb = MUTED

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("Trang ")
    add_page_number(footer)
    for run in footer.runs:
        run.font.size = Pt(9)
        run.font.color.rgb = MUTED


def add_bullets(doc, items):
    for item in items:
        paragraph = doc.add_paragraph(style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(4)
        paragraph.add_run(item)


def add_numbered(doc, items):
    for item in items:
        paragraph = doc.add_paragraph(style="List Number")
        paragraph.paragraph_format.space_after = Pt(4)
        paragraph.add_run(item)


def add_dataset_table(doc):
    data = [
        ("Video đầu vào", "dataset/videos/walking.mp4", "Nguồn demo chính"),
        ("Positive", "80 ảnh", "Crop người đi bộ từ video mẫu"),
        ("Negative", "120 ảnh", "Vùng nền ít overlap với người"),
        ("Test images", "5 ảnh", "Dùng xuất Gradient/HOG"),
        ("Custom model", "models/custom_hog_svm.pkl", "Linear SVM sau khi train"),
    ]
    table = doc.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    headers = ["Thành phần", "Số lượng / đường dẫn", "Vai trò"]
    for i, text in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = text
        set_cell_shading(cell, BLUE_GRAY)
        for run in cell.paragraphs[0].runs:
            run.bold = True
    set_repeat_table_header(table.rows[0])
    for row in data:
        cells = table.add_row().cells
        for i, text in enumerate(row):
            cells[i].text = text
    set_table_geometry(table, [1900, 3300, 4160])


def add_result_table(doc):
    data = [
        ("Train/Test split", "160 train, 40 test", "Stratified 80/20"),
        ("Test accuracy", "0.95", "Đánh giá trên tập test"),
        ("Test confusion matrix", "[[22, 2], [0, 16]]", "2 negative bị nhầm thành person"),
        ("Full dataset accuracy", "0.99", "Đánh giá lại trên 200 ảnh"),
        ("Full dataset matrix", "[[118, 2], [0, 80]]", "Không bỏ sót positive trong dataset"),
    ]
    table = doc.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    headers = ["Chỉ số", "Kết quả", "Ý nghĩa"]
    for i, text in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = text
        set_cell_shading(cell, BLUE_GRAY)
        for run in cell.paragraphs[0].runs:
            run.bold = True
    set_repeat_table_header(table.rows[0])
    for row in data:
        cells = table.add_row().cells
        for i, text in enumerate(row):
            cells[i].text = text
    set_table_geometry(table, [2300, 2600, 4460])


def add_code_block(doc, code):
    table = doc.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    set_table_geometry(table, [9360])
    cell = table.cell(0, 0)
    set_cell_shading(cell, "F7F7F7")
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(code)
    run.font.name = "Consolas"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Consolas")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Consolas")
    run.font.size = Pt(9)


def add_figure(doc, image_path, caption, width=5.7):
    path = ROOT / image_path
    if not path.exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(8)
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = MUTED


def build_report():
    doc = Document()
    configure_styles(doc)
    add_header_footer(doc)
    add_title_page(doc)

    doc.add_heading("1. Giới thiệu", level=1)
    doc.add_paragraph(
        "Đề tài xây dựng chương trình phát hiện người đi bộ trên video hoặc webcam "
        "bằng phương pháp xử lý ảnh truyền thống: Image Gradient, HOG và SVM. "
        "Hệ thống đọc từng frame, trích xuất đặc trưng hình dạng bằng HOG, dùng SVM "
        "để phân loại vùng ảnh, sau đó vẽ bounding box và nhãn trực tiếp lên luồng video."
    )
    add_bullets(doc, [
        "Demo realtime có bounding box, nhãn person, score và FPS.",
        "Có phần minh họa Sobel Gradient, Magnitude, Orientation và HOG visualization.",
        "Có phần tự tạo dataset positive/negative và train custom Linear SVM.",
    ])

    doc.add_heading("2. Cơ sở lý thuyết", level=1)
    doc.add_heading("2.1 Image Gradient", level=2)
    doc.add_paragraph(
        "Image Gradient mô tả sự thay đổi cường độ sáng theo không gian. Với ảnh xám I(x, y), "
        "gradient gồm hai thành phần Gx và Gy. Trong project, hai thành phần này được tính bằng "
        "toán tử Sobel."
    )
    add_code_block(doc, "Gx = dI/dx\nGy = dI/dy\nM(x, y) = sqrt(Gx^2 + Gy^2)\ntheta(x, y) = atan2(Gy, Gx)")

    doc.add_heading("2.2 HOG", level=2)
    doc.add_paragraph(
        "HOG, Histogram of Oriented Gradients, biểu diễn hình dạng đối tượng bằng histogram hướng gradient. "
        "Project dùng window 64x128, cell 8x8, block 2x2 cell và 9 hướng gradient."
    )
    add_numbered(doc, [
        "Resize vùng ảnh về 64x128 và chuyển sang grayscale.",
        "Tính gradient Gx, Gy, magnitude và orientation.",
        "Chia ảnh thành cell, tạo histogram hướng gradient cho từng cell.",
        "Gom cell thành block và chuẩn hóa L2-Hys.",
        "Ghép các block thành vector HOG cuối cùng.",
    ])

    doc.add_heading("2.3 SVM", level=2)
    doc.add_paragraph(
        "SVM nhận vector HOG và phân loại vùng ảnh thành person hoặc background. "
        "Với Linear SVM, hàm quyết định có dạng f(x) = w^T x + b. Nếu điểm quyết định lớn hơn "
        "ngưỡng, vùng ảnh được xem là có người."
    )

    doc.add_heading("3. Thiết kế chương trình", level=1)
    doc.add_paragraph("Luồng xử lý chính của demo:")
    add_numbered(doc, [
        "Đọc frame từ video hoặc webcam bằng cv2.VideoCapture.",
        "Resize frame để tăng tốc và áp dụng ROI nếu có.",
        "Dùng HOG + SVM để phát hiện người.",
        "Lọc bbox theo tỷ lệ, kích thước và Non-Maximum Suppression.",
        "Dùng tracker đơn giản để giảm nhấp nháy.",
        "Vẽ bounding box, nhãn, score và FPS lên frame.",
        "Hiển thị realtime hoặc ghi video kết quả.",
    ])

    doc.add_paragraph("Các module chính:")
    add_bullets(doc, [
        "src/detector.py: OpenCV HOG + SVM pretrained, NMS và vẽ kết quả.",
        "src/custom_detector.py: custom sliding-window detector.",
        "src/run_realtime.py: chạy webcam/video, ghi video.",
        "src/features.py: trích xuất HOG feature.",
        "src/bootstrap_dataset_from_video.py: tạo dataset từ video mẫu.",
        "src/train_custom_svm.py: train Linear SVM.",
        "src/visualize_gradient_hog.py: xuất ảnh Gradient/HOG.",
    ])

    doc.add_heading("4. Dataset và thí nghiệm", level=1)
    doc.add_paragraph(
        "Dataset được bootstrap từ video mẫu để bài nộp có đủ positive, negative và ảnh test. "
        "Cách này phù hợp cho demo học thuật; nếu mở rộng nghiêm túc hơn nên bổ sung dữ liệu chuẩn như INRIA Person Dataset."
    )
    add_dataset_table(doc)
    doc.add_paragraph("Lệnh tạo dataset:")
    add_code_block(doc, "python src/bootstrap_dataset_from_video.py --source dataset/videos/walking.mp4 --clear-generated")
    doc.add_paragraph("Lệnh train custom SVM:")
    add_code_block(doc, "python src/train_custom_svm.py")

    doc.add_heading("5. Kết quả", level=1)
    doc.add_paragraph(
        "Demo chính dùng OpenCV HOG + SVM pretrained để đảm bảo tốc độ và độ ổn định realtime. "
        "Custom SVM được train để chứng minh quy trình học máy truyền thống từ feature HOG."
    )
    add_result_table(doc)
    add_figure(doc, "outputs/input_resized.jpg", "Hình 1. Ảnh đầu vào sau khi resize.", 3.5)
    add_figure(doc, "outputs/gradient_magnitude.jpg", "Hình 2. Gradient magnitude.", 3.5)
    add_figure(doc, "outputs/gradient_orientation.jpg", "Hình 3. Gradient orientation.", 3.5)
    add_figure(doc, "outputs/hog_visualization.jpg", "Hình 4. HOG visualization.", 3.5)

    doc.add_heading("6. Nhận xét", level=1)
    doc.add_heading("Ưu điểm", level=2)
    add_bullets(doc, [
        "Phương pháp dễ giải thích và phù hợp nội dung xử lý ảnh truyền thống.",
        "Không cần GPU.",
        "Detector pretrained chạy ổn định trên video/webcam.",
        "Có custom model để thể hiện phần tự train SVM.",
    ])
    doc.add_heading("Hạn chế", level=2)
    add_bullets(doc, [
        "HOG + SVM nhạy với nền nhiều cạnh, che khuất và góc nhìn khác lạ.",
        "Custom SVM còn phụ thuộc vào độ đa dạng của dataset.",
        "Sliding window custom chậm hơn detector tối ưu sẵn của OpenCV.",
    ])

    doc.add_heading("7. Kết luận", level=1)
    doc.add_paragraph(
        "Project đáp ứng yêu cầu Chủ đề 2: trình bày được Image Gradient, HOG và SVM; "
        "xây dựng được demo phát hiện người đi bộ trên video/webcam; vẽ bounding box và nhãn realtime; "
        "đồng thời có phần tự tạo dataset, train custom Linear SVM và đánh giá kết quả."
    )

    doc.add_section(WD_SECTION.CONTINUOUS)
    doc.save(OUTPUT)
    print(f"Saved report: {OUTPUT}")


if __name__ == "__main__":
    build_report()
