import numpy as np
import matplotlib.pyplot as plt

try:
    K = int(input('Введите число K: '))
    N = int(input('Введите число N (размерность матрицы) кратное 2: '))
except ValueError:
    print('Ошибка неправильного ввода данных.\n'
          'Повторите попытку, необходимо ввести целочисленные данные.')
    exit()

if (N % 2 !=0) & (N<2):
    print('Введено неверное количество строк (столбцов) квадратной матрицы.\n')
    exit()

N = N // 2

# Генерация подматриц
B = np.random.randint(-10, 10, (N, N))
print('Подматрица b: \n', B, '\n')

C = np.random.randint(-10, 10, (N, N))
print('Подматрица c: \n', C, '\n')

D = np.random.randint(-10, 10, (N, N))
print('Подматрица d: \n', D, '\n')

E = np.random.randint(-10, 10, (N, N))
print('Подматрица e: \n', E, '\n')

A = np.vstack([np.hstack([B, C]), np.hstack([D, E])])
print('Матрица A: \n')
print(A, '\n')

F = A.copy()

countPlus = 0
countMinus = 0

for i in C:
    for j in i[1::2]:
        if j >= 0:
            countPlus += 1

for i in C:
    for j in i[::2]:
        if j < 0:
            countMinus += 1

if countPlus > countMinus:
    B1 = np.flip(B, axis=1)
    C1 = np.flip(C, axis=1)
    F = np.vstack([np.hstack([C1, B1]), np.hstack([D, E])])

else:
    F = np.hstack([np.vstack([B, D]), np.vstack([E, C])])


print('Матрица F: \n')
print(F, '\n')

print('Матрица A: \n')
print(A, '\n')

if np.linalg.det(A) > np.diagonal(F).sum():
    A_trans = np.transpose(A)
    F_inv = np.linalg.inv(F)
    result = A * A_trans - K * F_inv
else:
    A_inv = np.linalg.inv(A)
    F_inv = np.linalg.inv(F)
    G = np.tril(A)
    result = (A_inv + G - F_inv) * K

print('Результат вычисления выражения: \n')
print(result)

# Растровое представление
plt.subplot(2, 2, 1)
plt.imshow(F[:N, :N], cmap='rainbow', interpolation='bilinear')
plt.subplot(2, 2, 2)
plt.imshow(F[:N, N:], cmap='rainbow', interpolation='bilinear')
plt.subplot(2, 2, 3)
plt.imshow(F[N:, :N], cmap='rainbow', interpolation='bilinear')
plt.subplot(2, 2, 4)
plt.imshow(F[N:, N:], cmap='rainbow', interpolation='bilinear')
plt.show()

# Представление двумерных графиков
plt.subplot(2, 2, 1)
plt.plot(F[:N, :N])
plt.subplot(2, 2, 2)
plt.plot(F[:N, N:])
plt.subplot(2, 2, 3)
plt.plot(F[N:, :N])
plt.subplot(2, 2, 4)
plt.plot(F[N:, N:])
plt.show()