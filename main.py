import pandas as pd
import matplotlib.pyplot as plt


def loss_function(m, b, points):
    error = 0
    for i in range(len(points)):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        # Use MSE for the error:
        error += (y - (m*x + b))**2
    error = error / float(len(points))
    return error


def gradient_descent(m, b, points, learning_rate):
    m_grad = 0
    b_grad = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].studytime
        y = points.iloc[i].score

        # derivative w.r.t m:
        m_grad += -(2/n) * x * (y - (m*x + b))
        # derivative w.r.t b:
        b_grad += -(2/n) * (y - (m*x + b))

    new_m = m - m_grad * learning_rate
    new_b = b - b_grad * learning_rate
    return new_m, new_b

def main():
    data = pd.read_csv('study_scores.csv')

    m = 0
    b = 0
    learning_rate = 0.001
    rounds = 10000

    for i in range(rounds):
        if i % 50 == 0:
            print("loss: " + str(loss_function(m, b, data)))
        m, b = gradient_descent(m, b, data, learning_rate)
    
    print(m, b)
    plt.scatter(data.studytime, data.score, color="black")
    plt.plot(list(range(0, 20)), [m*x+b for x in range(0, 20)], color="red")
    plt.show()

main()