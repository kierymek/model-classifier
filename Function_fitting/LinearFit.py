from random import random
from matplotlib import pyplot as plt
import math

def linear_regression( X_data, Y_data ):

    sumX, sumX2, sumY, sumXY, a, b = 0, 0, 0, 0, 0, 0

    for i in range(len(X_data)):

        sumX = sumX + X_data[i]
        sumX2 = sumX2 + X_data[i] * X_data[i]
        sumY = sumY + Y_data[i]
        sumXY = sumXY + X_data[i] * Y_data[i]

    a = ((len(X_data)*sumXY) - (sumX*sumY)) / ((len(X_data)*sumX2) - (sumX*sumX))
    b = (sumY - a*sumX) / len(X_data)


    return [a, b]

def linear_genetic_optimization(X_data, Y_data):

    N = 50;
    number_of_steps = len(X_data)
    params = 2

    vec, vec2 = list(range(N * params)), list(range(N * params))
    cost, cost2 = list(range(N)), list(range(N))

    # agents initialization
    for i in range(N):
        vec[i] = 1 + ( random() -0.5)
        vec[i + N] = 10 * random()


    cost = calculate_cost(X_data, Y_data, vec);

    for loop in range(400):
        for j in range(N):
            e = 2 * random() - 1
            z1 = round((N - 1) * random())
            z2 = round((N - 1) * random())
            vec2[j] = vec[j] + e * (vec[z1] - vec[z2])
            vec2[j + N] = vec[j + N] + e * (vec[z1 + N] - vec[z2 + N])

            vec2[j] = apply_boundary(0, 3, vec2[j]);
            vec2[j + N] = apply_boundary(0, 10, vec2[j + N]);

        cost2 = calculate_cost(X_data, Y_data, vec2)

        for j in range(N):
            if cost2[j] < cost[j]:
                vec[j] = vec2[j]
                vec[j + N] = vec2[j + N]
                cost[j] = cost2[j]

    cost2 = calculate_cost(X_data, Y_data, vec)

    best_solution = cost2.index(min(cost2))

    return vec[best_solution], vec[best_solution + N]


def calculate_cost(X_data, Y_data, agents):
    Calculated_cost = 0
    cost = []
    N = int(len(agents)/2)
    for i in range(N):
        Calculated_cost = 0
        for k in range(len(X_data)):
            Calculated_cost = Calculated_cost + math.pow(X_data[k] * agents[i] + agents[i + N] - Y_data[k], 2);

        cost.append(Calculated_cost)

    return cost

def apply_boundary(min, max, value):
    result = 0
    if value < min :
        result = min + 0.01 * (max - min) * random()
    elif value > max :
        result = max - 0.01 * (max - min) * random()
    else :
        result = value

    return result


def main():

    X_data = [not_biased_x_point + 2 * (random() - 0.5)  for not_biased_x_point in range(50)]
    #Y = list(range(1,50))
    Y_data = [ (2 * x_point  + 2 + 10 * (random() - 0.5) ) for x_point in X_data]

    fig, ax = plt.subplots()
    ax.plot(X_data, Y_data, 'ro', label = "Data points")

    a, b = linear_regression(X_data, Y_data)
    print(a, b)
    X = list(range(50))
    Y = [ a * x_point + b for x_point in X]
    ax.plot(X, Y, 'green', label = "Linear regression")

    a_opt, b_opt = linear_genetic_optimization(X_data, Y_data)
    print(a_opt, b_opt)
    X_opt = list(range(50))
    Y_opt = [ a_opt * x_point + b_opt for x_point in X_opt]
    ax.plot(X_opt, Y_opt, 'blue', label = "Differential optimization")

    leg = ax.legend()

    a, b = round(a, 8), round(b, 8)
    a_opt, b_opt = round(a_opt, 8), round(b_opt, 8)
    ax.text(0.6, 0.2, "y = " + str(a) + "x + " + str(b), horizontalalignment='center',
         verticalalignment='center',
         transform=ax.transAxes, color = 'green')
    ax.text(0.6, 0.1, "y = " + str(a_opt) + "x + " + str(b_opt), horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes, color='blue')

    plt.show()

if __name__ == "__main__" :
    main()
