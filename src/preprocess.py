import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    input_path = "/opt/ml/processing/input"
    output_path = "/opt/ml/processing/output"

    # 读取数据（无数据集时自动生成模拟房价数据）
    data_path = os.path.join(input_path, "data.csv")
    if not os.path.exists(data_path):
        print("未检测到data.csv，生成模拟数据集")
        n_samples = 1000
        X1 = np.random.normal(50, 10, n_samples)
        X2 = np.random.normal(20, 5, n_samples)
        y = 2 * X1 + 3 * X2 + np.random.normal(0, 5, n_samples)
        df = pd.DataFrame({"X1": X1, "X2": X2, "y": y})
    else:
        df = pd.read_csv(data_path)

    # 划分训练集、测试集
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # 标准化
    scaler = StandardScaler()
    train.iloc[:, :-1] = scaler.fit_transform(train.iloc[:, :-1])
    test.iloc[:, :-1] = scaler.transform(test.iloc[:, :-1])

    # 保存输出
    train.to_csv(os.path.join(output_path, "train.csv"), index=False)
    test.to_csv(os.path.join(output_path, "test.csv"), index=False)
    print("预处理完成，已输出训练/测试集")
