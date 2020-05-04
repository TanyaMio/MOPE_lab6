import numpy as np
from scipy.stats import f, t
from tabulate import tabulate
import sklearn.linear_model as lm
import random

def mult(x1, x2, x3 = np.ones(15)):
    x = np.ones(N)
    for i in range(N):
        x[i] *= x1[i] * x2[i] * x3[i];
    return x


def f_x_func(x, b):
    global f_x
    f_x = np.zeros(N)
    for i in range(N):
        f_x[i] += b[0]
        for k in range(len(x[0])):
            f_x[i] += b[k + 1] * x[i][k]


def x_table(x_min, x_max):
  x01 = (x_max[0] + x_min[0]) / 2
  x02 = (x_max[1] + x_min[1]) / 2
  x03 = (x_max[2] + x_min[2]) / 2
  delta_x1 = x_max[0] - x01
  delta_x2 = x_max[1] - x02
  delta_x3 = x_max[2] - x03
  X1 = np.array([x_min[0], x_min[0], x_min[0], x_min[0], x_max[0], x_max[0], x_max[0], x_max[0], -l*delta_x1+x01, l*delta_x1+x01, x01, x01, x01, x01])
  X2 = np.array([x_min[1], x_min[1], x_max[1], x_max[1], x_min[1], x_min[1], x_max[1], x_max[1], x02, x02, -l*delta_x2+x02, l*delta_x2+x02, x02, x02])
  X3 = np.array([x_min[2], x_max[2], x_min[2], x_max[2], x_min[2], x_max[2], x_min[2], x_max[2], x03, x03, x03, x03, -l*delta_x3+x03, l*delta_x3+x03])
  return np.array(list(zip(X1, X2, X3, mult(X1, X2), mult(X1, X3), mult(X2, X3), mult(X1, X2, X3), mult(X1, X1), mult(X2, X2), mult(X3, X3))))


def y_val_table(x, b):
    y_val = np.zeros((N, m))
    f_x_func(x, b)
    for i in range(N):
        for j in range(m):
            y_val[i][j] += f_x[i] + random.random()*10 - 5
    return y_val


def calc_coef(X, y):
    x = list(X)
    for i in range(len(x)):
        x[i] = np.array([1, ] + list(x[i]))
    X = np.array(x)
    model = lm.LinearRegression(fit_intercept=False)
    model.fit(X, y)
    coefs = model.coef_
    print("\nThe regression equation:")
    print(f"y = {coefs[0]:.3f} + {coefs[1]:.3f} * x1 + {coefs[2]:.3f} * x2 + {coefs[3]:.3f} * x3"
          f" + {coefs[4]:.3f} * x1x2 + {coefs[5]:.3f} * x1x3 + {coefs[6]:.3f} * x2x3"
          f" + {coefs[7]:.3f} * x1x2x3 + {coefs[8]:.3f} * x1^2 + {coefs[9]:.3f} * x2^2 + {coefs[10]:.3f} * x3^2")
    return coefs


def eq_val_check(x_val, b_val):
    f_x_func(x_val, b_val)
    print("\n\nCheck:")
    for i in range(N):
        print(f"y{i+1:} = {b_val[0]:.3f} + {b_val[1]*x_val[i][0]:.3f} + {b_val[2]*x_val[i][1]:.3f}"
              f" + {b_val[3]*x_val[i][2]:.3f} + {b_val[4]*x_val[i][3]:.3f} + {b_val[5]*x_val[i][4]:.3f}"
              f" + {b_val[6]*x_val[i][5]:.3f} + {b_val[7]*x_val[i][6]:.3f} + {b_val[8]*x_val[i][7]:.3f}"
              f" + {b_val[9]*x_val[i][8]:.3f} + {b_val[10]*x_val[i][9]:.3f} = {f_x[i]:.3f}\t\ty_av{i+1:} = {y_mean[i]}")


def calc_disp(y, y_mean):
    disper = np.zeros(N)
    for i in range(N):
        for j in range(m):
            disper[i] += (y[i][j] - y_mean[i]) ** 2
        disper[i] /= m
    return disper


def matrix_print(y, x_list, y_mean, disper):
    global header_table
    header_table = ["№", "x1", "x2", "x3", "x1x2", "x1x3", "x2x3", "x1x2x3", "x1^2", "x2^2", "x3^2"]
    table = []
    for i in range(N):
        table.append([i + 1])
    for i in range(N):
        for _ in range(len(x_list[0])):
            table[i].append(x_list[i][_])
        for j in range(m):
            table[i].append(y[i][j])
        table[i].append(y_mean[i])
        table[i].append(disper[i])
    for i in range(m):
        header_table.append("Y" + str(i + 1))
    header_table.append("Y")
    header_table.append("S^2")
    print(tabulate(table, headers=header_table, tablefmt="fancy_grid"))


def print_eq(b):
    print((f"f(x1, x2, x3) = {b[0]:.1f} + x1 * {b[1]:.1f} + x2 * {b[2]:.1f} + x3 * {b[3]:.1f}"
           f" + x1x2 * {b[4]:.1f} + x1x3 * {b[5]:.1f} + x2x3 * {b[6]:.1f}"
           f" + x1x2x3 * {b[7]:.1f} + x1^2 * {b[8]:.1f} + x2^2 * {b[9]:.1f} + x3^2 * {b[10]:.1f}"))


def cohren(disper):
    global Gp, Gt, f1, f2
    Gp = max(disper) / sum(disper)
    f1 = m - 1
    f2 = N
    fisher = f.isf(*[q / f2, f1, (f2 - 1) * f1])
    Gt = round(fisher / (fisher + (f2 - 1)), 4)
    return Gp < Gt


def student_check():
    global sb, d, f3, t_exp
    d = len(x_code[0])
    f3 = f1 * f2
    sb = sum(disper) / N
    ssbs = sb / N * m
    sbs = ssbs ** 0.5
    beta = np.zeros(d)
    t_exp = []
    for j in range(d):
        for i in range(N):
            if (j == 0):
                beta[j] += y_mean[i]
            else:
                beta[j] += y_mean[i] * x_code[i][j]
        beta[j] /= N
        t_exp.append(abs(beta[j]) / sbs)

    ttabl = round(abs(t.ppf(q / 2, f3)), 4)
    print(f"\n\nAccording to Student's test:\ntcr = {ttabl:}")
    string_eq = f"y = {b[0]:.7f}"
    for i in range(len(t_exp)):
        if (t_exp[i] < ttabl):
            print(f"t{i:} = {t_exp[i]:.7f} = > Coefficient b{i:} is insignificant")
            b[i] = 0
            d = d - 1
        else:
            print(f"t{i:} = {t_exp[i]:.7f} = > Coefficient b{i:} is significant")
            if(i != 0): string_eq += f" + {b[i]:.7f} * " + header_table[i]
    print("\n\nThe regression equation now is:\n", string_eq)
    eq_val_check(x_list, b)


def fisher_check():
    global Fp, Ft
    f4 = N - d
    sad = 0
    for i in range(N):
        sad += (f_x[i] - y_mean[i]) ** 2
    sad *= (m / (N - d))
    Fp = sad / sb
    print(f"\n\nFp = {Fp:}", end="\t\t\t")
    Ft = round(abs(f.isf(q, f4, f3)), 4)
    print(f"Ft = {Ft:}")

x_min = np.array([-40, -35, 20])
x_max = np.array([20, 15, 25])
b_initial = np.array([2.2, 1.6, 9.2, 9.5, 0.2, 0.9, 8.7, 9.1, 0.8,0.7, 6.5])
m = 2
k = 3
p = 0
N = 14
l = k**(1/2)
q = 0.05
x_code = x_table(np.array([-1, -1, -1]), np.array([1, 1, 1]))
x_list = x_table(x_min, x_max)
y = y_val_table(x_list, b_initial)
y_mean = np.sum(y, axis=1) / m
disper = calc_disp(y, y_mean)
cohren(disper)
while not cohren(disper):
    m = m + 1
    y = y_val_table(x_list, b_initial)
    y_mean = np.sum(y, axis=1) / m
    disper = calc_disp(y, y_mean)
print("Normalized Experiment:")
matrix_print(y, x_code, y_mean, disper)
print("\n\nNaturalized Experiment:")
matrix_print(y, x_list, y_mean, disper)
print(f"\nm = {m:} \nGp = {Gp:}\t\t\tGt = {Gt:}\nGp < Gt -> According to Cochran's C-test homogeneity of variance is confirmed with probability of 0.95")
b = calc_coef(x_list, y_mean)
eq_val_check(x_list, b)
student_check()
fisher_check()
if Fp > Ft:
    print("Fp > Ft = > According to Fisher's F-test model is not adequate to the original with probability of 0.95.")
else:
    print("Fp < Ft = > According to Fisher's F-test model is adequate to the original with probability of 0.95.")
