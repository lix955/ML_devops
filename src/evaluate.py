import os
import json
import joblib
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

if __name__ == "__main__":
    model_path = "/opt/ml/processing/model/model.joblib"
    test_path = "/opt/ml/processing/test/test.csv"
    output_path = "/opt/ml/processing/evaluation"

    # 加载模型和测试集
    model = joblib.load(model_path)
    df = pd.read_csv(test_path)
    X_test = df.drop("y", axis=1)
    y_test = df["y"]

    y_pred = model.predict(X_test)

    # 计算指标
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # 输出评估报告（流水线会读取这个json）
    evaluation_report = {
        "accuracy": r2,
        "mse": mse
    }

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "evaluation.json"), "w") as f:
        json.dump(evaluation_report, f)

    print(f"评估完成 R2:{r2:.4f} MSE:{mse:.4f}")
