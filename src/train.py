import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    train_path = "/opt/ml/input/data/train/train.csv"
    model_dir = "/opt/ml/model"

    # 加载训练数据
    df = pd.read_csv(train_path)
    X_train = df.drop("y", axis=1)
    y_train = df["y"]

    # 训练简单回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 保存模型
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    print("模型训练完成并保存")
