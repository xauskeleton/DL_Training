# Lộ trình project thực hành Deep Learning

## Đọc trước khi bắt đầu

Lộ trình gồm 10 project nhỏ, đi từ MLP, CNN, phát hiện đối tượng (Faster R-CNN, YOLO), ViT, nén mô hình (pruning, quantization) đến fine-tune một model ngôn ngữ nhỏ (Qwen2.5) bằng LoRA cho bài tóm tắt văn bản, và kết thúc bằng hai project luyện kỹ năng research. Người học đã đọc đủ lý thuyết; các project này biến những gì đã đọc thành trực giác có số liệu. Project sau dùng lại code và model của project trước, nên cần làm theo thứ tự.

Quy tắc chung cho mọi project:

- **Một repo GitHub duy nhất**, mỗi project một thư mục, có README ghi cách chạy lại và kết quả chính.
- **Được dùng AI để code. Điều được kiểm tra là mức độ hiểu.** Người học phải giải thích được mọi dòng trong code nộp. Người hướng dẫn có thể chỉ vào một dòng bất kỳ và hỏi: "bỏ hoặc đổi dòng này thì chuyện gì xảy ra?"
- **Dùng dữ liệu nhỏ.** Mọi project chạy trên tập con nhỏ của bộ dữ liệu gốc để chạy nhanh trên Kaggle. Tiêu chí "Đạt khi" tính trên tập con đó. Cách lấy tập con (lấy những lớp nào, bao nhiêu mẫu, có cân bằng giữa các lớp không) phải cố định bằng seed và ghi trong README.
- **Dự đoán trước khi chạy.** Trước mỗi thí nghiệm, ghi một dòng: kết quả kỳ vọng và lý do. Chạy xong, so với dự đoán. Chỗ lệch chính là chỗ cần học. Dự đoán sai không bị trừ điểm; không dự đoán mới là vấn đề.
- **Số seed:** run nhẹ (MNIST, tập con CIFAR, model nhỏ) chạy 3 seed, báo cáo mean ± std. Run nặng (detection, ViT pretrained, fine-tune Qwen) chạy 1 seed.
- **Early stopping cho mọi bài train.** Theo dõi chỉ số trên tập validation (không bao giờ dùng tập test), dừng khi chỉ số không cải thiện sau số epoch (hoặc số lần đánh giá) bằng patience, giữ checkpoint tốt nhất. Báo cáo ghi rõ dừng ở epoch nào. Tập test chỉ dùng một lần ở cuối.
- **Phải chạy inference.** Mỗi project có script `infer.py` nhận đầu vào mới ngoài bộ dữ liệu (ảnh tự chụp, ảnh từ internet, bài báo mới), trả về dự đoán, có hình minh hoạ, và đo thời gian suy luận. Kết quả inference đưa vào báo cáo.
- **Chỉ số đánh giá và hyperparams** dùng theo mục "Chỉ số đánh giá" và "Hyperparams gợi ý" của từng project. Mỗi bộ dữ liệu có chỉ số chuẩn riêng, phải báo cáo đúng chỉ số đó. Hyperparams gợi ý là điểm xuất phát thông dụng; đổi thì ghi lý do.
- **Debug là việc của mọi project, không phải một bài riêng.** Code AI sinh ra hay chạy được nhưng sai ngầm, nên trước mỗi thí nghiệm thật luôn chạy checklist sanity check: overfit một batch nhỏ về loss ≈ 0, loss ban đầu ≈ ln(số lớp) với bài phân loại, xem trực tiếp vài mẫu và nhãn sau DataLoader (với detection: vẽ box lên ảnh), kiểm tra `model.train()`/`model.eval()` và `zero_grad()`. Checklist nằm trong README và được bổ sung mỗi khi gặp lỗi mới. Người hướng dẫn có thể cài lỗi im lặng vào code của bất kỳ project nào; người học phải tự tìm ra mà không so với bản gốc. Tài liệu tham khảo: bài "A Recipe for Training Neural Networks" của Karpathy.
- **Log bằng TensorBoard hoặc Weights & Biases.** Mọi run phải có đường loss và metric.
- **Báo cáo cho mỗi project** (có thể vài trang), gồm:
    - **Dữ liệu:** bộ dữ liệu gốc là gì, tập con lấy thế nào, có những lớp nào, số mẫu mỗi lớp, cách chia train/val/test, vài mẫu minh hoạ. Với detection thêm số object mỗi lớp và phân bố kích thước box.
    - Đã làm gì, cấu hình chính.
    - Bảng và biểu đồ kết quả.
    - Các dự đoán sai và vì sao sai.
    - Những điều còn chưa hiểu.
- **Máy chạy:** Kaggle (2×T4) là đủ cho cả 10 project.

Mỗi project gồm: mục tiêu, việc cần làm, chỉ số đánh giá, hyperparams gợi ý, inference, đạt khi nào, và câu hỏi kiểm tra. Chưa đạt tiêu chí thì chưa chuyển sang project tiếp theo.

---

## Giai đoạn 1 — Nền tảng: MLP và CNN

Hết giai đoạn này, người học có một bộ khung train chuẩn của riêng mình và hiểu vì sao CNN hợp với ảnh.

### P1 — Build lại MLP bằng PyTorch

**Dữ liệu gợi ý:** MNIST.

**Mục tiêu:** hiểu các thành phần cơ bản của PyTorch đang làm thay việc gì, bằng cách build cùng một MLP hai lần: lần đầu bằng tensor + autograd, lần sau bằng thư viện `nn`.

**Việc cần làm:**

1. **Bản tensor + autograd:** build MLP 2 lớp ẩn (ReLU, softmax, cross-entropy) chỉ bằng tensor PyTorch, train bằng mini-batch SGD. Trọng số tự khai báo với `requires_grad=True`, gradient lấy từ `loss.backward()`, tự cập nhật trọng số bằng một dòng SGD.
2. **Bản thư viện PyTorch:** build lại bằng `nn.Module`, `nn.Linear`, `nn.CrossEntropyLoss`, `torch.optim.SGD` và `DataLoader`.
3. Với cùng siêu tham số, so accuracy và thời gian train của hai bản.
4. Thử 3 cách khởi tạo trên bản thư viện: toàn số 0, ngẫu nhiên quá lớn, mặc định của `nn.Linear`. Dự đoán trước, rồi vẽ loss của cả ba.

**Chỉ số đánh giá:** accuracy trên tập test, confusion matrix, đường loss train/val.

**Hyperparams gợi ý:** lớp ẩn 256 và 128; SGD lr 0.1 (hoặc 0.01 kèm momentum 0.9); batch 128; tối đa 30 epoch; early stopping theo val accuracy, patience 5. Tách 10% tập train làm validation.

**Inference:** viết `infer.py` nhận một ảnh chữ số, trả về nhãn và xác suất. Chạy trên khoảng 10 chữ số người học tự viết tay rồi chụp lại (đưa về 28×28, nền đen chữ trắng như MNIST), ghi lại các ca sai.

**Đạt khi:** cả hai bản đều đạt test accuracy ≥ 97% và cho kết quả tương đương nhau; chỉ ra được mỗi đoạn code của bản tensor + autograd ứng với thành phần nào trong bản thư viện; giải thích được vì sao khởi tạo bằng 0 không học.

**Câu hỏi kiểm tra:**

- `autograd` làm gì khi gọi `loss.backward()`? Gradient được lưu ở đâu?
- Vì sao phải xoá gradient cũ trước mỗi bước? Quên thì chuyện gì xảy ra?
- `nn.Linear` chứa những tham số nào và mặc định khởi tạo ra sao?
- `optimizer.zero_grad()` và `optimizer.step()` tương ứng với dòng nào trong bản tensor + autograd?

### P2 — CNN: training loop và ablation

**Dữ liệu gợi ý:** tập con CIFAR-10, ví dụ 10.000 ảnh train (1.000 ảnh mỗi lớp), giữ nguyên tập test.

**Mục tiêu:** hiểu CNN làm gì khác MLP, có một bộ khung train chuẩn dùng lại cho các project sau, và tự thấy tác dụng của những thứ đã đọc về training dynamics.

**Việc cần làm:**

1. Dựng bộ khung (được dùng AI để code, nhưng phải đọc hiểu từng phần): Dataset/DataLoader, training loop, eval loop, cố định seed, lưu/nạp checkpoint, đọc cấu hình từ file YAML hoặc argparse. Không dùng Lightning hay Keras.
2. Train MLP của P1 trên dữ liệu này, rồi train một CNN nhỏ (vài lớp conv + pooling) có số tham số tương đương. Dự đoán trước chênh lệch.
3. Làm thử mixed precision để biết (`torch.autocast` + `GradScaler`): so với FP32 về thời gian mỗi epoch, VRAM và accuracy. Đây chỉ là bước tìm hiểu; **mọi thí nghiệm trong lộ trình đều chạy FP32** (ngoại lệ duy nhất là bản Qwen 1.5B tuỳ chọn ở P8).
4. Train ResNet-18 (bản cho ảnh 32×32) và làm ablation, mỗi lần chỉ đổi một yếu tố, 3 seed mỗi cấu hình:
    - SGD + momentum so với AdamW
    - LR cố định so với warmup + cosine
    - Có và không có BatchNorm
    - Có và không có residual connection
    - Có và không có data augmentation (random crop + flip)
5. Tìm LR tốt bằng cách quét 5 giá trị theo thang log.

**Chỉ số đánh giá:** top-1 accuracy, accuracy từng lớp, confusion matrix.

**Hyperparams gợi ý:** ResNet-18 với SGD lr 0.1, momentum 0.9, weight decay 5e-4, cosine; AdamW lr 1e-3, weight decay 0.05; batch 128; tối đa 100 epoch; early stopping theo val accuracy, patience 10. CNN nhỏ và MLP: Adam lr 1e-3, tối đa 50 epoch. Tách 10% tập con train làm validation.

**Inference:** `infer.py` nhận một ảnh bất kỳ, đưa về 32×32, trả về top-3 lớp kèm xác suất. Chạy trên khoảng 20 ảnh lấy ngoài CIFAR (ảnh tự chụp hoặc từ internet), đo thời gian suy luận mỗi ảnh.

**Đạt khi:** ResNet-18 với cấu hình tốt nhất đạt ≥ 80% test accuracy; có bảng so sánh MLP và CNN nhỏ; có bảng ablation mean ± std; người khác clone repo chạy lại được bằng một lệnh. Checkpoint ResNet-18 tốt nhất được giữ lại để dùng ở P6 và P7.

**Câu hỏi kiểm tra:**

- Vì sao CNN ít tham số hơn mà lại tốt hơn MLP trên ảnh? Weight sharing và local connectivity nghĩa là gì?
- `model.train()` và `model.eval()` khác nhau ở đâu với BatchNorm và Dropout?
- Bỏ residual connection thì ResNet-18 gặp vấn đề gì khi train?
- Mixed precision làm gì, và vì sao dùng float16 thì cần `GradScaler`?
- Chênh lệch giữa hai cấu hình nhỏ hơn std thì kết luận được gì?

---

## Giai đoạn 2 — Phát hiện đối tượng: Faster R-CNN và YOLO

Hết giai đoạn này, người học hiểu bài toán detection từ dữ liệu, cách đánh giá, đến hai họ phương pháp two-stage và one-stage. P3 và P4 dùng chung một tập dữ liệu để so sánh trực tiếp.

**Dữ liệu:** bộ African Wildlife của Ultralytics: 1.504 ảnh động vật hoang dã, 4 lớp (trâu rừng, voi, tê giác, ngựa vằn), chia sẵn 1.052 train, 225 val, 227 test. Nhãn có sẵn định dạng YOLO nên dùng thẳng cho P4; với P3 chỉ cần viết một hàm đổi nhãn YOLO (tâm, rộng, cao, chuẩn hoá) sang box pixel (x1, y1, x2, y2) cho torchvision. Bộ này nhỏ và ít lớp nên train nhanh hơn nhiều so với Pascal VOC.

### P3 — Faster R-CNN

**Mục tiêu:** hiểu một detector two-stage chạy thế nào: anchor, RPN, RoI Align, NMS, và cách tính mAP.

**Việc cần làm:**

1. **Hiểu dữ liệu:** đọc annotation, vẽ box lên ảnh, thống kê số object mỗi lớp, số object mỗi ảnh và phân bố kích thước box. Phần này đưa vào mục Dữ liệu của báo cáo.
2. **Tự viết các hàm cốt lõi:** IoU, NMS, và tính AP cho một lớp. So kết quả với `torchvision.ops.box_iou`, `torchvision.ops.nms` và công cụ đánh giá chuẩn (pycocotools hoặc torchmetrics).
3. **Fine-tune:** dùng Faster R-CNN ResNet-50-FPN của torchvision (pretrained COCO), fine-tune trên bộ dữ liệu. Đánh giá mAP@0.5 và mAP@[0.5:0.95]. Nếu vẫn chậm, thay bằng bản Faster R-CNN MobileNetV3-Large-FPN của torchvision (nhẹ hơn nhiều) và ghi rõ trong báo cáo.
4. **Ablation**, dự đoán trước rồi mới chạy:
    - Số proposal giữ lại sau RPN: 1000, 300, 50
    - Kích thước anchor: mặc định so với chỉ anchor lớn
    - Đóng băng backbone so với fine-tune toàn bộ
    - Ngưỡng NMS: 0.3, 0.5, 0.7

**Chỉ số đánh giá:** mAP@0.5 (chỉ số kiểu Pascal VOC), mAP@[0.5:0.95] (chỉ số chuẩn kiểu COCO), AP từng lớp, recall, FPS.

**Hyperparams gợi ý:** SGD lr 0.005, momentum 0.9, weight decay 5e-4; batch 4; cạnh ngắn ảnh 512; tối đa 15 epoch, giảm lr 10 lần ở 2/3 quá trình; early stopping theo val mAP@0.5, patience 3. Dùng tập val có sẵn để early stopping.

**Inference:** `infer.py` nhận ảnh, vẽ box, nhãn và điểm tin cậy lên ảnh, có tham số ngưỡng điểm (ví dụ 0.5). Chạy trên khoảng 20 ảnh tự chụp hoặc lấy ngoài bộ dữ liệu, đo FPS.

**Đạt khi:** IoU, NMS, AP tự viết khớp với thư viện; model fine-tune đạt mAP@0.5 ≥ 70%; có bảng ablation kèm giải thích.

**Câu hỏi kiểm tra:**

- "Two-stage" nghĩa là gì? Stage 1 và stage 2 mỗi bên học cái gì?
- RPN gán nhãn dương/âm cho anchor theo quy tắc nào?
- RoI Align khác RoI Pooling ở đâu, vì sao quan trọng với box nhỏ?
- Vì sao cần NMS? NMS có thể làm mất object đúng trong trường hợp nào?
- mAP@0.5 và mAP@[0.5:0.95] khác nhau thế nào, cái nào khắt khe hơn?

### P4 — YOLO: build lại để hiểu, rồi train bằng Ultralytics

**Mục tiêu:** hiểu detector one-stage bằng cách tự build một bản YOLO thu nhỏ, rồi biết dùng công cụ thực tế (Ultralytics) và so sánh với Faster R-CNN.

**Việc cần làm:**

1. **Build mini YOLO** theo ý tưởng YOLOv1: backbone ResNet-18 pretrained, chia ảnh thành lưới S×S, mỗi ô dự đoán box, objectness và lớp. Tự viết phần gán nhãn cho ô lưới, hàm loss (box + objectness + lớp), phần giải mã output và NMS (dùng lại NMS ở P3).
2. Train mini YOLO trên cùng bộ dữ liệu với P3.
3. **Phá từng thành phần**, dự đoán trước rồi mới chạy:
    - Bỏ objectness loss
    - Đổi kích thước lưới (ví dụ 7×7 so với 14×14)
    - Đổi trọng số giữa loss box và loss lớp
4. **Train bằng Ultralytics:** train **YOLOv5m** và **YOLOv8m** trên cùng bộ dữ liệu (nhãn đã sẵn định dạng YOLO). Đọc file cấu hình model và các tham số train (augmentation, mosaic, lịch LR), ghi lại mỗi tham số chính làm gì, và chỉ ra khác biệt kiến trúc giữa v5 và v8.
5. **So sánh 4 model:** mini YOLO, YOLOv5m, YOLOv8m, Faster R-CNN ở P3. So mAP, tốc độ suy luận (FPS trên T4), số tham số. Phân tích lỗi riêng trên object nhỏ.

**Chỉ số đánh giá:** mAP@0.5, mAP@[0.5:0.95], precision, recall (Ultralytics tính sẵn), AP từng lớp, FPS. Mini YOLO tính bằng đúng hàm đánh giá ở P3 để so sánh công bằng.

**Hyperparams gợi ý:**
- Mini YOLO: ảnh 448×448, lưới 7×7, λ_coord = 5, λ_noobj = 0.5 (theo YOLOv1); Adam lr 1e-4; batch 16; tối đa 60 epoch; early stopping theo val mAP@0.5, patience 10.
- YOLOv5m, YOLOv8m: khởi tạo từ trọng số pretrained COCO; `imgsz=512`, `batch=16`, `epochs=60`, `patience=10` (early stopping có sẵn của Ultralytics); các tham số còn lại để mặc định.

**Inference:** chạy cả 4 model trên cùng khoảng 20 ảnh ngoài bộ dữ liệu, xuất ảnh có box, ghép thành bảng so sánh trực quan. Với Ultralytics dùng `model.predict`. Đo FPS ở batch 1.

**Đạt khi:** mini YOLO phát hiện được object trên ảnh test, mAP@0.5 ≥ 40%; có bảng so sánh 4 model; giải thích được vì sao các model khác nhau về tốc độ và về độ chính xác trên object nhỏ.

**Câu hỏi kiểm tra:**

- Vì sao YOLO nhanh hơn Faster R-CNN? Đánh đổi lại cái gì?
- Một object được gán cho ô lưới nào, và chuyện gì xảy ra khi hai object rơi vào cùng một ô?
- YOLOv5 dùng anchor, YOLOv8 thì không. Anchor-based và anchor-free khác nhau thế nào, ảnh hưởng gì đến kết quả trên bộ dữ liệu này?
- Mosaic augmentation giúp gì, nhất là khi dữ liệu ít?
- Vì sao object nhỏ khó với cả hai họ detector?

---

## Giai đoạn 3 — Vision Transformer

### P5 — ViT: build lại và so với CNN

**Dữ liệu gợi ý:** tập con CIFAR-10 của P2 cho phần train từ đầu; tập con CIFAR-100 (ví dụ 100 ảnh mỗi lớp) cho phần pretrained.

**Mục tiêu:** hiểu ViT xử lý ảnh khác CNN thế nào, và khi nào ViT thắng, khi nào thua.

**Việc cần làm:**

1. **Build mini ViT từ đầu:** patch embedding, CLS token, position embedding, các khối Transformer encoder. Train từ đầu trên tập con CIFAR-10, so với ResNet-18 ở P2 với cùng số epoch. Dự đoán trước bên nào thắng.
2. **Phá và đổi**, dự đoán trước rồi mới chạy:
    - Patch size 4 so với 8
    - Bỏ position embedding
    - Thay CLS token bằng average pooling
3. **Dùng model pretrained:** fine-tune ViT-B/16 (thư viện `timm`) và ResNet-50 pretrained trên tập con CIFAR-100, ở 2 mức dữ liệu: 100 ảnh mỗi lớp và 10 ảnh mỗi lớp. Với ViT, so thêm linear probe và full fine-tune.
4. Vẽ attention map (attention rollout) của ViT pretrained trên vài ảnh.

**Chỉ số đánh giá:** top-1 accuracy; với CIFAR-100 thêm top-5 accuracy; số tham số và thời gian mỗi epoch.

**Hyperparams gợi ý:**
- Mini ViT train từ đầu: patch 4, chiều 192, 6 lớp, 3 head, MLP ratio 4; AdamW lr 1e-3, weight decay 0.05; batch 128; warmup 5 epoch rồi cosine; tối đa 150 epoch; early stopping theo val accuracy, patience 15.
- Fine-tune pretrained (ảnh 224): full fine-tune AdamW lr 1e-4, linear probe lr 1e-3; batch 32; tối đa 20 epoch; early stopping patience 3.

**Inference:** `infer.py` cho ViT pretrained, nhận ảnh, trả về top-5 lớp kèm attention map chồng lên ảnh. Chạy trên khoảng 10 ảnh ngoài bộ dữ liệu.

**Đạt khi:** có bảng so sánh ViT và CNN khi train từ đầu và khi dùng pretrained; có hình attention map; giải thích được khác biệt giữa hai trường hợp.

**Câu hỏi kiểm tra:**

- CNN có sẵn những "giả định" (inductive bias) nào về ảnh mà ViT không có?
- Vì sao ViT cần position embedding?
- Chi phí attention tăng thế nào khi giảm patch size?
- CLS token làm gì?

---

## Giai đoạn 4 — Nén mô hình

Hai project này làm trên ResNet-18 đã train ở P2, để kết quả dễ đọc và dễ so sánh.

### P6 — Pruning trên CNN: L1, Taylor, Fisher

**Mục tiêu:** hiểu các tiêu chí đánh giá độ quan trọng của kênh, cách chọn lớp để cắt, và khác biệt giữa "nhỏ hơn trên giấy" và "nhanh hơn thật".

**Chọn lớp nào để cắt, và tại sao:**

- **Không cắt lớp conv đầu tiên:** ít kênh, nhận ảnh gốc, cắt vào là mất thông tin ngay từ đầu.
- **Không cắt lớp FC cuối:** số đầu ra bằng số lớp, cố định.
- **Không cắt các conv có đầu ra đi vào phép cộng residual** (conv thứ hai trong mỗi BasicBlock và nhánh downsample): đầu ra của chúng phải cùng số kênh với nhánh tắt, cắt một chỗ là phải cắt đồng bộ cả chuỗi residual.
- **Chỉ cắt conv thứ nhất trong mỗi BasicBlock:** số kênh đầu ra của lớp này chỉ dùng bên trong block, cắt kênh của nó thì chỉ cần sửa đầu vào của conv thứ hai ngay sau.
- Kiểm chứng lựa chọn bằng phân tích độ nhạy (bước 1) và ghi rõ lý do trong báo cáo.

**Việc cần làm:**

1. **Độ nhạy từng lớp:** với các lớp được phép cắt, cắt riêng từng lớp ở mức cao (ví dụ 70% kênh), không fine-tune, xem lớp nào làm accuracy tụt mạnh nhất. Dùng kết quả này để quyết định tỉ lệ cắt cho từng lớp.
2. **Cắt kênh theo 3 tiêu chí**, cùng các lớp và cùng tỉ lệ cắt (ví dụ 30%, 50%, 70% kênh):
    - **L1:** độ quan trọng của kênh = tổng trị tuyệt đối trọng số của filter.
    - **Taylor:** độ quan trọng = ước lượng bậc một của mức tăng loss khi bỏ kênh, tính từ activation × gradient trên vài batch.
    - **Fisher:** độ quan trọng = trung bình bình phương gradient theo kênh (xấp xỉ Fisher information) trên vài batch.
3. Với mỗi tiêu chí và mỗi tỉ lệ, đo accuracy trước và sau fine-tune lại.
4. Đo số tham số, FLOPs và tốc độ suy luận thật trên GPU và CPU của model đã cắt so với model gốc.

Dùng thư viện Torch-Pruning hoặc tự viết các tiêu chí. Taylor và Fisher cần vài batch dữ liệu để tính gradient; ghi rõ dùng bao nhiêu batch.

**Chỉ số đánh giá:** accuracy và mức tụt so với model gốc; số tham số; FLOPs; độ trễ (ms/ảnh ở batch 1) và thông lượng (ảnh/giây ở batch 64) trên GPU và CPU; tỉ lệ nén.

**Hyperparams gợi ý:** Taylor và Fisher tính trên 20 batch, mỗi batch 64 ảnh; fine-tune sau khi cắt: SGD lr 0.01, momentum 0.9, cosine, tối đa 30 epoch; early stopping theo val accuracy, patience 5.

**Inference:** chạy `infer.py` của P2 với model gốc và các model đã cắt trên cùng một bộ ảnh, so dự đoán và đo độ trễ thật (khởi động vài lượt trước khi đo, lấy trung bình nhiều lượt).

**Đạt khi:** có biểu đồ độ nhạy từng lớp; có bảng accuracy theo tỉ lệ cắt cho cả 3 tiêu chí, trước và sau fine-tune; có bảng tham số, FLOPs và tốc độ thật; báo cáo giải thích được vì sao chọn những lớp đó và vì sao các tiêu chí cho kết quả khác nhau.

**Câu hỏi kiểm tra:**

- L1 chỉ nhìn trọng số, còn Taylor và Fisher nhìn cả dữ liệu. Ưu và nhược của mỗi cách?
- Taylor và Fisher khác nhau ở đâu, dù cả hai đều dùng gradient?
- Vì sao không cắt các conv có đầu ra đi vào phép cộng residual?
- Cắt kênh ở conv thứ nhất của block thì lớp nào phía sau phải sửa theo?
- Pruning xong không fine-tune lại thì accuracy tụt ở mức nào, vì sao?

### P7 — Quantization: FP32, FP16, INT8

**Mục tiêu:** biết dùng các công cụ có sẵn để giảm độ chính xác số của model, và thấy đánh đổi giữa kích thước, tốc độ và accuracy.

**Việc cần làm** (dùng thư viện có sẵn, không tự cài đặt thuật toán):

1. **FP32:** đo accuracy, kích thước file và tốc độ suy luận của ResNet-18 gốc trên GPU và CPU.
2. **FP16:** chuyển model sang float16, đo lại trên GPU.
3. **INT8 (PTQ):** dùng static post-training quantization của `torch.ao.quantization` với cấu hình mặc định, khoảng 100 ảnh để calibration. Đo lại trên CPU.
4. Lập bảng so sánh 3 bản: accuracy, kích thước file, độ trễ.
5. **(Tuỳ chọn)** Export YOLOv8m ở P4 sang FP16 và INT8 bằng chức năng export của Ultralytics, so mAP và tốc độ.

**Chỉ số đánh giá:** accuracy và mức tụt so với FP32; kích thước file (MB); độ trễ ms/ảnh ở batch 1 (FP32 và INT8 trên CPU, FP32 và FP16 trên GPU).

**Hyperparams gợi ý:** calibration 100 ảnh lấy từ tập train; cấu hình quantization mặc định của thư viện cho CPU x86 (`fbgemm`/`x86`).

**Inference:** chạy cùng một bộ ảnh qua 3 bản FP32, FP16, INT8; đếm số ảnh mà các bản dự đoán khác nhau; đo độ trễ theo cùng cách ở P6.

**Đạt khi:** có bảng FP32, FP16, INT8 với đủ 3 chỉ số; giải thích được vì sao mỗi bản nhanh hơn hoặc chậm hơn trên phần cứng tương ứng.

**Câu hỏi kiểm tra:**

- FP32, FP16 và INT8 khác nhau thế nào về cách biểu diễn số?
- Vì sao INT8 cần dữ liệu calibration?
- Vì sao FP16 chạy nhanh trên GPU nhưng thường không nhanh hơn trên CPU?
- Accuracy của INT8 tụt nhiều thì nên thử gì tiếp?

---

## Giai đoạn 5 — NLP với LoRA

### P8 — Fine-tune Qwen2.5 nhỏ bằng LoRA cho bài tóm tắt tin tức tiếng Việt

**Bài toán:** đầu vào là một bài báo tiếng Việt, đầu ra là đoạn tóm tắt (text → text).

**Dữ liệu:** bộ VietNews trên Hugging Face ([harouzie/vietnews](https://huggingface.co/datasets/harouzie/vietnews)), khoảng 99 nghìn bài train, 22 nghìn validation, 22 nghìn test; mỗi mẫu có `title`, `abstract` (tóm tắt), `article` (bài báo). Lấy tập con nhỏ: khoảng 3.000 bài train, 300 validation, 500 test. Bỏ các bài quá dài, hoặc cắt bài báo về khoảng 512–768 token.

**Model:** Qwen2.5-0.5B-Instruct (chạy FP32 trên một T4 được). Muốn thử Qwen2.5-1.5B-Instruct thì đây là ngoại lệ duy nhất dùng FP16: trọng số gốc FP16, adapter LoRA FP32, `fp16=True` trong `Trainer`. T4 không hỗ trợ BF16.

**Mục tiêu:** biết fine-tune model sinh văn bản bằng LoRA, biết đánh giá output dạng text, và hiểu LoRA đổi lại được gì so với fine-tune toàn bộ.

**Việc cần làm:**

1. **Chuẩn bị dữ liệu:** viết prompt theo chat template của Qwen ("Tóm tắt bài báo sau: ..."), tokenize, đặt nhãn sao cho loss chỉ tính trên phần tóm tắt, không tính trên phần bài báo. Thống kê độ dài bài báo và tóm tắt theo token.
2. **Baseline:** chạy model gốc chưa fine-tune (zero-shot) trên tập test, tính ROUGE-1, ROUGE-2, ROUGE-L.
3. **LoRA:** fine-tune bằng `transformers` + `peft`. Thử rank 4, 8, 16 và hai cách gắn: chỉ vào `q_proj`, `v_proj` so với vào mọi lớp linear.
4. **Full fine-tune** model 0.5B để so sánh.
5. Ghi lại cho mỗi cấu hình: ROUGE, số tham số được train, VRAM, thời gian train. Đọc tay khoảng 20 bản tóm tắt, ghi các lỗi hay gặp (bịa thông tin, lặp câu, chép nguyên câu đầu bài).
6. Gộp (merge) trọng số LoRA vào model gốc, kiểm tra output sinh ra không đổi.

**Chỉ số đánh giá:**
- ROUGE-1, ROUGE-2, ROUGE-L, chỉ số chuẩn của bài tóm tắt. Lưu ý: tokenizer mặc định của thư viện `rouge_score` bỏ mọi ký tự ngoài a–z và 0–9, nên mất hết dấu tiếng Việt. Phải tách từ theo khoảng trắng (hoặc truyền tokenizer riêng) trước khi tính.
- Loss trên tập validation; độ dài trung bình của tóm tắt sinh ra.

**Hyperparams gợi ý:**
- LoRA: lr 2e-4; `lora_alpha` = 2 × rank; `lora_dropout` 0.05.
- Full fine-tune: lr 2e-5.
- Chung: batch hiệu dụng 16 (batch 2 × gradient accumulation 8); warmup 3%, cosine; tối đa 3 epoch; độ dài tối đa 1.024 token; gradient checkpointing.
- Early stopping theo val loss: đánh giá mỗi 100 bước, `EarlyStoppingCallback` với patience 3, giữ checkpoint tốt nhất.
- Sinh văn bản: greedy decoding, `max_new_tokens=128`.

**Inference:** `infer.py` nhận một bài báo tiếng Việt, trả về tóm tắt. Chạy trên khoảng 10 bài báo mới lấy ngoài bộ dữ liệu, so tóm tắt của model gốc và model đã fine-tune, đo thời gian sinh mỗi bài.

**Đạt khi:** LoRA vượt rõ baseline zero-shot về ROUGE; có bảng so sánh đủ các cấu hình; có phần phân tích lỗi từ các mẫu đọc tay; merge LoRA cho output giống trước khi merge.

**Câu hỏi kiểm tra:**

- Vì sao loss chỉ nên tính trên phần tóm tắt? Tính trên cả bài báo thì sao?
- LoRA thêm gì vào một lớp linear? Vì sao khởi tạo B = 0?
- Vì sao LoRA không làm chậm lúc suy luận sau khi merge?
- ROUGE đo được gì và không đo được gì về chất lượng tóm tắt?
- Greedy decoding và sampling cho kết quả khác nhau thế nào trên bài này?

---

## Giai đoạn 6 — Kỹ năng research

Hai project cuối luyện cách làm research: tái hiện kết quả của người khác, tự tra cứu để nắm tiêu chí đánh giá nghiên cứu, nơi công bố, viết tổng quan một mảng mới, và viết paper. Các quy tắc chung (dữ liệu nhỏ, early stopping, inference, số seed) vẫn áp dụng.

### P9 — Tái hiện paper Cutout

**Paper:** DeVries & Taylor (2017), "Improved Regularization of Convolutional Neural Networks with Cutout". Repo chính thức: [uoguelph-mlrg/Cutout](https://github.com/uoguelph-mlrg/Cutout).

**Ý tưởng:** khi train, che ngẫu nhiên một ô vuông trên mỗi ảnh đầu vào (gán giá trị 0), buộc model không dựa vào một vùng duy nhất của ảnh.

**Số trong paper (ResNet-18, CIFAR-10 đầy đủ, trung bình 5 lần chạy, có flip và dịch ảnh):** test error 4,72% khi không dùng Cutout, 3,99% khi dùng Cutout với ô vuông cạnh 16 pixel.

**Mục tiêu:** tự cài lại được phương pháp của một paper, và biết kiểm tra code của mình bằng cách đối chiếu với repo chính thức của tác giả.

**Dữ liệu:** tập con CIFAR-10 của P2 (10.000 ảnh train, giữ nguyên tập test).

**Việc cần làm:**

1. **Chạy repo của tác giả:** sửa phần nạp dữ liệu để dùng đúng tập con của P2, thêm early stopping. Chạy ResNet-18 có và không có Cutout, theo lệnh trong README (`python train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16`). Ghi lại các chỗ phải sửa và lỗi gặp khi cài đặt.
2. **Tự code Cutout** trong bộ khung P2, chỉ dựa trên paper. Chỉ mở repo của tác giả để đối chiếu khi bị kẹt, và ghi lại những chỗ phải xem repo mới hiểu (ví dụ ô vuông có được phép tràn ra ngoài ảnh không, che trước hay sau khi chuẩn hoá).
3. **Kiểm tra bản tự code:** vẽ khoảng 16 ảnh sau Cutout của cả hai bản để so bằng mắt.
4. **So sánh hai bản** trên cùng dữ liệu, cùng hyperparams, 3 seed mỗi bên: test accuracy có và không có Cutout.
5. **Ablation kích thước ô vuông:** cạnh 0, 8, 12, 16, 20 pixel. Dự đoán trước kích thước tốt nhất trên tập con nhỏ.
6. **Inference:** chạy `infer.py` của P2 với model có và không có Cutout trên cùng bộ ảnh ngoài CIFAR, đặc biệt các ảnh bị che khuất một phần.

**Chỉ số đánh giá:** test accuracy và test error (paper báo cáo bằng error), mean ± std qua 3 seed.

**Hyperparams gợi ý:** theo repo: SGD lr 0.1, momentum 0.9, weight decay 5e-4, nesterov; batch 128; giảm lr theo mốc. Repo train 200 epoch trên dữ liệu đầy đủ; trên tập con dùng early stopping theo val accuracy, patience 15.

**Sản phẩm nộp:** repo có bản tự code; bảng so sánh số của paper, số từ repo tác giả và số từ bản tự code; bảng ablation kích thước ô vuông; danh sách chỗ paper không nói rõ.

**Đạt khi:** bản tự code và bản repo cho kết quả chênh nhau trong mức nhiễu giữa các seed; Cutout cải thiện so với không dùng Cutout trên tập con, hoặc giải thích được vì sao không; có bảng ablation.

**Câu hỏi kiểm tra:**

- Cutout giống và khác dropout ở điểm nào?
- Vì sao paper cho phép ô vuông tràn ra ngoài biên ảnh?
- Mức cải thiện trên tập con nhỏ lớn hơn hay nhỏ hơn trên dữ liệu đầy đủ, vì sao?
- Kích thước ô vuông tốt nhất phụ thuộc vào điều gì?

### P10 — Tự research: tiêu chí đánh giá, hội nghị và tạp chí, tổng quan distillation và continual learning, cách viết paper

**Mục tiêu:** biết tự tra cứu để nắm bức tranh chung của giới nghiên cứu AI: một nghiên cứu được đánh giá theo tiêu chí nào, công bố ở đâu, tự viết được tổng quan về một mảng mới ở mức nhập môn, và biết cách viết một paper.

**Việc cần làm:**

1. **Tra cứu tiêu chí đánh giá một nghiên cứu khoa học.** Tìm hướng dẫn cho reviewer của các hội nghị lớn (ví dụ NeurIPS, ICLR, CVPR) và checklist khả năng tái hiện của NeurIPS. Tóm tắt các tiêu chí họ dùng (thường gồm tính mới, ý nghĩa, độ chặt chẽ, khả năng tái hiện, trình bày) và thang điểm reviewer chấm.
2. **Tra cứu các hội nghị và tạp chí lớn trong AI theo từng mảng:**
    - Học máy nói chung, thị giác máy tính, xử lý ngôn ngữ tự nhiên, trí tuệ nhân tạo nói chung, và các mảng khác nếu thấy cần.
    - Với mỗi nơi ghi: tên, mảng, hội nghị hay tạp chí, xếp hạng (ví dụ theo CORE, h5-index của Google Scholar, hoặc phân hạng Q của Scimago với tạp chí), chu kỳ hạn nộp trong năm, tỉ lệ chấp nhận nếu tìm được.
    - Ghi nguồn cho mọi con số.
3. **Viết tổng quan knowledge distillation** (vài trang): bài toán là gì, ý tưởng teacher và student, các nhóm phương pháp chính, ứng dụng, các vấn đề còn mở. Chỉ cần đọc ở mức survey, bài blog, bài giảng, không cần đọc sâu từng paper.
4. **Viết tổng quan continual learning** (vài trang): bài toán là gì, các kịch bản (task-, domain-, class-incremental), các nhóm phương pháp chính (regularization, replay, kiến trúc, dùng model pretrained), chỉ số đánh giá, các vấn đề còn mở. Mức đọc như bước 3.
5. Trong hai bản tổng quan, chỉ ra chỗ hai mảng gặp nhau (distillation được dùng để chống quên trong continual learning).
6. **Học cách viết một paper:**
    - Tra cứu các hướng dẫn viết paper khoa học (ví dụ bài giảng "How to Write a Great Research Paper" của Simon Peyton Jones) và hướng dẫn trình bày của chính các hội nghị ở bước 2.
    - Tóm tắt cấu trúc chuẩn của một paper AI và mỗi phần cần trả lời câu hỏi gì: tiêu đề, abstract, giới thiệu (vấn đề, đóng góp), related work, phương pháp, thí nghiệm (thiết lập, kết quả chính, ablation), kết luận, hạn chế.

**Sản phẩm nộp:** bản tóm tắt tiêu chí đánh giá kèm link nguồn; bảng hội nghị và tạp chí theo mảng kèm nguồn; tổng quan distillation; tổng quan continual learning; bản tóm tắt cách viết paper.

**Đạt khi:** mọi thông tin tra cứu đều có nguồn; bảng hội nghị và tạp chí chia rõ theo mảng; mỗi bản tổng quan nhóm được các phương pháp theo cách tiếp cận, người chưa biết gì đọc xong hiểu được mảng đó giải quyết vấn đề gì; bản tóm tắt cách viết paper nêu được mỗi phần của paper cần trả lời câu hỏi gì.

**Câu hỏi kiểm tra:**

- Reviewer của một hội nghị AI lớn chấm bài theo những tiêu chí nào?
- Vì sao trong AI hội nghị lại quan trọng ngang hoặc hơn tạp chí?
- Distillation chuyển "kiến thức" gì từ teacher sang student?
- Task-incremental, domain-incremental và class-incremental khác nhau thế nào?
- Vì sao distillation dùng được để chống quên?
- Abstract và phần giới thiệu của một paper cần trả lời những câu hỏi nào?
- Phần hạn chế (limitations) nên viết gì, vì sao các hội nghị yêu cầu có phần này?

---

## Theo dõi tiến độ

Người học tự cập nhật cột Trạng thái (Chưa bắt đầu / Đang làm / Đạt) và cột Kết quả chính. Chỉ chuyển sang "Đạt" khi đủ tiêu chí ở mục "Đạt khi" của project đó.

| Project | Trạng thái | Tài liệu / công cụ chính | Kết quả chính |
| --- | --- | --- | --- |
| P1 — Build lại MLP bằng PyTorch | Chưa bắt đầu | Hướng dẫn "Learn the Basics" của PyTorch | |
| P2 — CNN, training loop + ablation | Chưa bắt đầu | d2l.ai (bản PyTorch), chương CNN và ResNet | |
| P3 — Faster R-CNN | Chưa bắt đầu | Paper Faster R-CNN (Ren và cộng sự, 2015); hướng dẫn detection của torchvision | |
| P4 — YOLO: mini YOLO + YOLOv5m, YOLOv8m | Chưa bắt đầu | Paper YOLOv1 (Redmon và cộng sự, 2016); tài liệu Ultralytics | |
| P5 — ViT | Chưa bắt đầu | Paper ViT (Dosovitskiy và cộng sự, 2020); thư viện `timm` | |
| P6 — Pruning: L1, Taylor, Fisher | Chưa bắt đầu | Thư viện Torch-Pruning; paper Taylor pruning (Molchanov và cộng sự, 2019) | |
| P7 — Quantization: FP32, FP16, INT8 | Chưa bắt đầu | `torch.ao.quantization` | |
| P8 — Qwen2.5 + LoRA tóm tắt tin tức | Chưa bắt đầu | Paper LoRA (Hu và cộng sự, 2021); `transformers`, `peft`; bộ VietNews | |
| P9 — Tái hiện paper Cutout | Chưa bắt đầu | Paper Cutout (DeVries & Taylor, 2017); repo `uoguelph-mlrg/Cutout` | |
| P10 — Tự research: tiêu chí, hội nghị và tạp chí, tổng quan distillation và continual learning, cách viết paper | Chưa bắt đầu | Hướng dẫn reviewer của NeurIPS/ICLR/CVPR; CORE; Google Scholar Metrics | |
