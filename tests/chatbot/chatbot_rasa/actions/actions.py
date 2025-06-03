# Logic trả lời theo custom action


import os
from typing import Any, Dict, List, Text

import joblib
import pandas as pd
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionDuBaoKhoiLuong(Action):
    def name(self) -> Text:
        return "action_du_bao_khoi_luong"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        # Load dữ liệu đã chuẩn bị
        data_path = "assets/processed/model_data.csv"
        model_path = "models/volume_predictor.pkl"

        if not os.path.exists(data_path) or not os.path.exists(model_path):
            dispatcher.utter_message(
                text="Hiện tại chưa có dữ liệu hoặc mô hình để dự báo."
            )
            return []

        df = pd.read_csv(data_path)
        model = joblib.load(model_path)

        latest_features = df.iloc[[-1]].drop(columns=["Date", "Volume"])
        y_pred = model.predict(latest_features)[0]

        dispatcher.utter_message(
            text=f"Dự báo khối lượng giao dịch ngày mai là khoảng {int(y_pred):,} đơn vị."
        )
        return []
