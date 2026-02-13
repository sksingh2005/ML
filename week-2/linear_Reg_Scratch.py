def linear_regression(x, y):
    n = len(x)
    
    sum_x = 0
    sum_y = 0
    sum_xy = 0
    sum_x2 = 0
    
    for i in range(n):
        sum_x += x[i]
        sum_y += y[i]
        sum_xy += x[i] * y[i]
        sum_x2 += x[i] * x[i]
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_x2 - sum_x * sum_x
    m = numerator / denominator
    
    b = (sum_y - m * sum_x) / n
    
    return m, b

cgpa = [7, 8, 6.5, 7, 9]
ctc = [20, 30, 50, 40, 50]

m, b = linear_regression(cgpa, ctc)

print("Slope (m):", m)
print("Intercept (b):", b)

def predict(x_value, m, b):
    return m * x_value + b

x=float(input("Enter CGPA to predict CTC: "))
print("Prediction:", predict(x, m, b))
