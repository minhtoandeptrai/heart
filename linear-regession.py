import numpy as np

def gradient_descent(X, y, learning_rate, num_iterations):
  # Khởi tạo w0 và w1
  w0 = 22
  w1 = 5.5
  m = len(X)    
  gap = 0

 
  for i in range(800):
    # Tính giá trị dự đoán
    y_predicted = w0 + w1 * X

    # Tính MSE
    mse = (1/m) * np.sum((y_predicted - y)**2)
  
    # Tính gradient
    dw0 = (1/m) * np.sum(y_predicted - y)
    dw1 = (1/m) * np.sum((y_predicted - y) * X)

    # Cập nhật tham số
    w0 = w0 - learning_rate * dw0
    w1 = w1 - learning_rate * dw1

    print(f"step {i} MSE = {mse:.4f} dw0:{dw0:.4f} dw1:{dw1:.4f} W0:{w0:.4f}  W1:{w1:.4f} ")
    i = i + 1
  return w0, w1

# Ví dụ sử dụng
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([30, 32, 35, 37, 40, 43, 45, 47, 50, 53])

learning_rate = 0.01
num_iterations = 1000

w0, w1 = gradient_descent(X, y, learning_rate, num_iterations)
print("Final w0:", w0)
print("Final w1:", w1)