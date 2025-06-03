from Model_Predict.Pipeline_predict import predict_volume
import sys

def predictor():
    models = ["linear_regression.pkl", "polynomial_regression.pkl", "random_forest.pkl", "xgb_model.json", "knn.pkl"]
    print("1. Linear Regression\n2. Polymial Regression\n3. Random Forest\n4. XGBoost\n5. KNN")
    choose = int(input("chọn model dùng để dự đoán: "))
    try:
        model = models[choose-1]
    except:
        print("Vui lòng chọn từ 1 đén 5")
        sys.exit()

    date = input("Nhập ngày cần dự đoán (YYYY-MM-DD): ")

    predicted = predict_volume(model_file=model, date_target=date, history_window=30, timedelta_days=60)
    print(f"\nKết quả trả về của predict_volume: {predicted}")

    if predicted.get('real_volume') is None:
        print(f"\nVolume dự đoán ngày {predicted['date']} là {predicted['predicted_volume']}")
    else:
        print(f"\nVolume dự đoán ngày {predicted['date']} là: {predicted['predicted_volume']}")
        print(f"Volume thực tế là {predicted['real_volume']} với sai số: {predicted['residual']}")

if __name__=="__main__":
    predictor()