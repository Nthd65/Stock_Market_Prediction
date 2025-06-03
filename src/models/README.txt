## File model_predict này chạy trên python 3.10.11 ##

Để dùng được hàm predict_volume, cần import như sau: ** from Model_Predict.Pipeline_predict import predict_volume **
với điều kiện fodler Model_Predict nằm chung folder với file code

##################### Hàm chính ########################

predict_volume(model_file, date_target, history_window=30, timedelta_days=50)
	
	trong đó:
		model_file (str): file của mô hình đã huấn luyện dùng để dự báo.
		date_target (str): ngày cần dự báo theo định dạng "YYYY-MM-DD".

		history_window (int, mặc định 30 - 1 tháng): số ngày lịch sử dùng để tính các chỉ báo đặc trưng cho dự báo.

		timedelta_days (int, mặc định 60 - 2 tháng): số ngày lùi về trước so với date_target để lấy dữ liệu làm tiền xử lý trước khi đưa vào model.


--- Nếu gặp phải ValueError: Không đủ dữ liệu thô để tính window khởi tạo ---
		HÃY TĂNG CHỈ SỐ timedelta_days LÊN

################ Test Hàm predict_volume ################

-> Chạy file Predictor.py
-> chọn model muốn sử dụng
-> chọn ngày muốn mô hình dự đoán

kết quả: predict_volume sẽ trả về một dictionary
		Nếu ngày cần dự báo là quá khứ -> {'date': (date), 'predicted_volume': (volume được dự đoán),
							'real_volume': (volume thật sự), 'residual': (sai số)}

		Nếu ngày cần dự đoán là tương lai hoặc hôm nay -> {'date': (date), 
								   'predicted_volume': (volume được dự đoán)}

***Chú ý: dự đoán ngày hiện tại và tương lai sẽ trả về cảnh báo do hàm get_data trong pipeline_predict in ra***
	Muốn tắt, tìm get_data trong pipeline_predict.py chỉnh warning=False


################# Ví dụ kết quả của Predictor.py #################

- Ở đây sẽ chọn model Random Forest và dự đoán ngày 2025-5-29 (quá khứ)

--------------------------------------------------------------------------------------

1. Linear Regression
2. Polymial Regression
3. Random Forest
4. XGBoost
5. KNN
chọn model dùng để dự đoán: 3
Nhập ngày cần dự đoán (YYYY-MM-DD): 2025-5-29
YF.download() has changed argument auto_adjust default to True
[*********************100%***********************]  1 of 1 completed

Kết quả trả về của predict_volume: {'date': '2025-05-29', 'predicted_volume': 20738, 'real_volume': 20943, 'residual': 205}

Volume dự đoán ngày 2025-05-29 là: 20738
Volume thực tế là 20943 với sai số: 205

Process finished with exit code 0

----------------------------------------------------------------------------------------



- Tiếp tục chọn model Random Forest và dự đoán ngày 2025-6-15 (tương lai)

 --------------------------------------------------------------------------------------

1. Linear Regression
2. Polymial Regression
3. Random Forest
4. XGBoost
5. KNN
chọn model dùng để dự đoán: 3
Nhập ngày cần dự đoán (YYYY-MM-DD): 2025-6-15
YF.download() has changed argument auto_adjust default to True
[*********************100%***********************]  1 of 1 completed
***CẢNH BÁO: Ngày cuối cùng trong data lấy được là 2025-05-29, KHÔNG trùng với ngày yêu cầu (2025-06-15)***

Kết quả trả về của predict_volume: {'date': '2025-06-15', 'predicted_volume': 5655}

Volume dự đoán ngày 2025-06-15 là 5655

Process finished with exit code 0

---------------------------------------------------------------------------------------
