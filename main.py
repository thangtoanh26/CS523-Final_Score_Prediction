import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)

    w1 = model.coef_[0]      # Hệ số góc
    w0 = model.intercept_    # Tung độ gốc

    return model, w1, w0

def plot_data_and_model(X, y, model, w1, w0):
    plt.figure(figsize=(10, 6))
    
    plt.scatter(X, y, color='blue', alpha=0.6, label='Dữ liệu thực tế (TRAIN2.csv)')
    
    y_pred = model.predict(X)
    
    plt.plot(X, y_pred, color='red', linewidth=2, label='Đường hồi quy dự báo ($y = w_1x + w_0$)')
    
    plt.title('Mô hình Hồi quy tuyến tính: Dự đoán Điểm cuối kỳ', fontsize=14, fontweight='bold')
    plt.xlabel('Điểm giữa kỳ (Midterm)', fontsize=12)
    plt.ylabel('Điểm cuối kỳ (Final)', fontsize=12)
    
    formula_text = f'Final = {w1:.2f} * Midterm + {w0:.2f}'
    plt.text(min(X), max(y), formula_text, fontsize=12, color='red', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    print("Đang hiển thị đồ thị... (Hãy đóng cửa sổ đồ thị để tiếp tục)")
    plt.show(block=True)

def main():
    file_path = 'TRAIN2.csv'
    
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file '{file_path}'. Vui lòng đặt file này cùng thư mục với script.")
        return

    print("Đang đọc dữ liệu...")
    df = pd.read_csv(file_path)
    
    df = df.dropna(subset=['midterm', 'final'])
    
    X = df['midterm'].values.reshape(-1, 1)
    y = df['final'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24521613)

    print("Đang huấn luyện mô hình...")
    model, w1, w0 = train_model(X_train, y_train)

    print("\n================================================================")
    print("KẾT QUẢ HUẤN LUYỆN MÔ HÌNH")
    print("================================================================")
    print(f"Hệ số góc (w1)     : {w1:.4f}")
    print(f"Tung độ gốc (w0)   : {w0:.4f}")
    print(f"Công thức dự đoán   : Final = {w1:.4f} * Midterm + ({w0:.4f})")
    print(f"Độ chính xác mô hình: {model.score(X_test, y_test):.4f} (R² Score)")
    print("================================================================\n")

    plot_data_and_model(X, y, model, w1, w0)

    print("\n--- CÔNG CỤ DỰ ĐOÁN NHANH ---")
    print("Nhập 'q' hoặc 'quit' để thoát chương trình.")
    while True:
        user_input = input("Nhập điểm giữa kỳ của sinh viên: ")
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("Kết thúc chương trình. Tạm biệt!")
            break
        
        try:
            midterm_score = float(user_input)
            if midterm_score < 0 or midterm_score > 10:
                print("Cảnh báo: Điểm thường nằm trong khoảng từ 0 đến 10.")
                
            predicted_final = model.predict([[midterm_score]])[0]
            
            predicted_final = max(0.0, min(10.0, predicted_final))
            
            print(f"  -> Điểm cuối kỳ dự báo: {predicted_final:.2f}\n")
        except ValueError:
            print("  Lỗi: Vui lòng nhập một số hợp lệ!\n")

if __name__ == "__main__":
    main()
