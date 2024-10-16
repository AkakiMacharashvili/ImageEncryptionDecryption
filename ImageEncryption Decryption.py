import math
import cv2
import numpy



#task1
def image_from_path_to_T(path):
    image = cv2.imread(path)
    array = numpy.array(image)
    return array

def string_from_input_to_T():
    str = input('please enter string: ')
    return str


#task2

#for integer entries
def text_from_T_to_X(text):
    lst = []
    for a in text:
        order = ord(a)
        lst.append([order])
    return lst



def image_from_T_to_X(image):
    n, m, t = numpy.shape(image)
    vector = []
    for i in range(n):
        for j in range(m):
            for k in range(t):
                vector.append([image[i][j][k]])
    vector.append([n])
    return vector


# from PIL import Image as im
#
# def image_from_T_to_path(array):
#     data = im.fromarray(array)
#     data.save('from_X_to_T.png')

#for float entries
def text_from_T_to_X_float(text):
    lst = []
    for a in text:
        order = ord(a)
        lst.append(order)
    lst1 = []
    for i in lst:
        lst1.append([float(i)])
    return lst1

def image_from_T_to_X_float(image):
    n, m, t = numpy.shape(image)
    vector = []
    for i in range(n):
        for j in range(m):
            for k in range(t):
                vector.append([float(image[i][j][k])])
    vector.append([n])
    return vector



#task3

def k_int(n):
    lst = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append((i + j) % 7)
        lst.append(row)
    for i in range(n):
        lst[i][i] = n * 15

    return lst

def encription(vector):
    K_of_text = numpy.array(k_int(len(vector)))
    Y_for_text = numpy.matmul(K_of_text, vector)
    return Y_for_text

def vector_len(vector):
    tot = 0
    for a in vector:
        tot += a*a
    vector_length = math.sqrt(tot)
    return vector_length

def projection(vec1, vec2):
    numerator = 0
    for i in range(len(vec1)):
        numerator += vec1[i] * vec2[i]

    denominator = 0
    for i in range(len(vec1)):
        denominator += vec1[i] * vec1[i]

    if denominator == 0:
        return []
    scalar = float(numerator / denominator)

    vector = numpy.multiply(vec1, scalar)

    return vector

def gramm_schmit(v):
    n = len(v)
    u = []
    first = numpy.array(v[0])
    u.append(first)
    for i in range(1, n):
        v_cur = v[i]
        next_vec = v_cur
        for j in range(i):
            l = projection(u[j], v[i])
            if numpy.array_equal(l, []):
                return False
            next_vec -= projection(u[j], v[i])
        u.append(next_vec)
    e = []
    for a in u:
        scalar = vector_len(a)
        #tolerance
        if scalar <= math.pow(10, -15):
            return False
        e.append(numpy.divide(a, scalar))

    st = set()
    for i in e:
        st.add(tuple(i))

    if len(st) != n or len(st) != len(v[0]):
        return False
    return True

#task4
def convert_binary(n):
    max = math.pow(2, 31)
    binary = []
    while max >= 1:
        if n >= max:
            n -= max
            binary.append(1)
        else:
            binary.append(0)
        max = max / 2
    binary.reverse()
    return binary

def convert_binary_float(n):
    cur = 0.5
    binary = []
    for i in range(32):
        if n >= cur:
            n -= cur
            binary.append(1)
        else:
            binary.append(0)
        cur = cur * 0.5


    return binary

def convert_decimal_float(binary):
    cur = 0.5
    decimal = 0.0
    for i in binary:
        decimal += cur * i
        cur = cur * 0.5
    return decimal

def convert_decimal(binary):
    decimal = 0
    for i in range(len(binary)):
        if(binary[i]):
            decimal += pow(2, i)
    return decimal

def hide_image_int(mat1, mat2):
    n = len(mat1)
    mat2[0][0] = n
    for i in range(n):
        binary = convert_binary(mat1[i][0])
        for j in range(0, len(binary), 2):
            k = int(j / 2 + 1)
            bit0 = binary[j]
            bit1 = binary[j + 1]
            decimal = mat2[16*i + k][0]
            decimal_array = convert_binary(decimal)
            decimal_array[0] = bit0
            decimal_array[1] = bit1
            decimal_num = convert_decimal(decimal_array)
            mat2[16*i + k][0] = decimal_num
    return mat2

def hide_image_float(mat1, mat2):
    n = len(mat1)
    mat2[0][0] = 2 * n
    for i in range(n):
        binary1 = convert_binary(int(mat1[i][0]))
        binary2 = convert_binary_float((mat1[i][0] % 1))
        # print(binary2)
        for j in range(0, len(binary1), 2):
            k = int(j / 2 + 1)
            bit0 = binary1[j]
            bit1 = binary1[j + 1]
            decimal = mat2[32*i + k][0]
            decimal_array = convert_binary(decimal)
            decimal_array[0] = bit0
            decimal_array[1] = bit1
            decimal_num = convert_decimal(decimal_array)
            mat2[32*i + k][0] = decimal_num

        for j in range(0, len(binary2), 2):
            k = int(j / 2 + 1) + 16
            bit0 = binary2[j]
            bit1 = binary2[j + 1]
            decimal = mat2[32*i + k][0]
            decimal_array = convert_binary_float(decimal)
            decimal_array[0] = bit0
            decimal_array[1] = bit1
            decimal_num = convert_decimal(decimal_array)
            mat2[32*i + k][0] = decimal_num
    return mat2




#task5

def convert_back_float(mat2):

    decimal_lst = []
    for i in range(0, mat2[0][0], 2):
        converter_array = []
        converter_array_float = []
        for j in range(16):
            converter_array.append(mat2[16*i + j + 1][0])
        for j in range(16):
            converter_array_float.append(mat2[16*(i + 1) + j + 1][0])
        binary_lst = []
        for a in converter_array:
            binary = convert_binary(a)
            binary_lst.append(binary[0])
            binary_lst.append(binary[1])

        float_list = []
        for a in converter_array_float:
            binary = convert_binary(a)
            float_list.append(binary[0])
            float_list.append(binary[1])

        decimal = convert_decimal(binary_lst)
        float_num = convert_decimal_float(float_list)
        decimal_lst.append(decimal + float_num)
    return decimal_lst

def convert_back_int(mat2):
    decimal_lst = []
    for i in range(0, mat2[0][0]):
        converter_array = []
        for j in range(16):
            converter_array.append(mat2[16 * i + j + 1][0])
        binary_lst = []
        for a in converter_array:
            binary = convert_binary(a)
            binary_lst.append(binary[0])
            binary_lst.append(binary[1])
        decimal = convert_decimal(binary_lst)
        decimal_lst.append(decimal)
    return decimal_lst


#task6
def inf_norm(vector1, vector2):
    mx = 0
    for i in range(len(vector1)):
        temp = vector1[i] - vector2[i]
        temp = abs(temp)
        mx = max(temp, mx)
    return mx

def Gauss_Seidel(K, Y):
    x = []
    n = len(K)
    max_iteration = 100
    for i in range(n):
        x.append(1)
    iter = 0
    cond = True
    x_next = []
    tol = math.pow(10, -15)
    while iter < max_iteration and cond:
        iter += 1
        x_next = []
        for i in range(n):
            sum1 = 0
            sum2 = 0
            for j in range(i):
                sum1 += K[i][j] * x_next[j]
            for j in range(i + 1, n):
                sum2 += K[i][j] * x[j]
            cur = (Y[i][0] - sum1 - sum2) / K[i][i]
            x_next.append(cur)

        if inf_norm(x_next, x) < tol:
            cond = False
        x = x_next
    return x_next

def Jacob(K, Y):
    x = []
    n = len(K)
    max_iteration = 100
    for i in range(n):
        x.append(1)
    iter = 0
    cond = True
    x_next = []
    tol = math.pow(10, -8)
    while iter < max_iteration and cond:
        iter += 1
        x_next = []
        for i in range(n):
            sum = 0
            for j in range(n):
                if i != j:
                   sum += x[j] * K[i][j]
            cur = (Y[i][0] - sum) / K[i][i]
            x_next.append(cur)

        if inf_norm(x_next, x) < tol:
            cond = False
        x = x_next
    return x_next

def Richardson(A, b):
    n = len(A)
    m = len(A[0])
    number_of_iterations = 1000
    tol = math.pow(10, -15)
    P = []
    inverse_of_P = []
    for i in range(n):
        lst = []
        lst1 = []
        for j in range(m):
            lst.append(0)
            lst1.append(0.)
        P.append(lst)
        inverse_of_P.append(lst1)
    for i in range(n):
        tot = 0
        for j in range(m):
            tot += abs(A[i][j])
        P[i][i] = tot
        inverse_of_P[i][i] = float(1/tot)

    x = []
    for i in range(m):
        x.append([0])

    cond = True
    iter = 0
    while cond and iter <= number_of_iterations:
        iter += 1
        Ax = numpy.matmul(A, x)
        b_Ax = numpy.subtract(b, Ax)
        matrix = numpy.add(x, b_Ax)
        next_x = numpy.matmul(inverse_of_P, matrix)
        if inf_norm(next_x, x) < tol:
            cond = False
        x = next_x
    return x

def vector_sclar_multiplication(vector1, vector2):
    tot = 0

    for i in range(len(vector1)):

        tot += vector1[i][0] * vector2[i][0]
    return tot

def Non_Stationary_Method(A, b):
    n = len(A)
    x = []
    for i in range(n):
        x.append([i])
    number_of_iteration = 100
    tol = math.pow(10, -15)
    iter = 0
    cond = True
    while cond and iter <= number_of_iteration:
        iter += 1
        Ax = numpy.matmul(A, x)
        r = numpy.subtract(b, Ax)
        Ar = numpy.matmul(A, r)
        numerator = vector_sclar_multiplication(Ar, r)
        denominator = vector_sclar_multiplication(Ar, Ar)



        if denominator == 0:
            return []
        alfa = numerator / denominator
        ar = numpy.multiply(r, alfa)
        next_x = numpy.add(x, ar)
        if inf_norm(next_x, x) <= tol:
            cond = False
        x = next_x
    return x

def decript_text(Y_of_text):
    n = len(Y_of_text)
    K = k_int(n)
    I = []
    for i in range(n):
        lst = []
        for j in range(n):
            if i == j:
                lst.append(1)
            else:
                lst.append(0)
        I.append(lst)

    X = Gauss_Seidel(K, Y_of_text)

    return X

def decript_image(Y_of_image):
    Y_of_image[0][0] = Y_of_image[0][0] % 255
    n = len(Y_of_image)
    K = k_int(n)
    I = []
    for i in range(n):
        lst = []
        for j in range(n):
            if i == j:
                lst.append(1)
            else:
                lst.append(0)
        I.append(lst)
    X = Gauss_Seidel(K, Y_of_image)
    return X

#task7

def text_for_X_to_T_int(vector):
    text = ''
    for a in vector:
        text += chr(a)
    return text

def image_from_X_to_T_int(vector):
    image = []
    n = vector[len(vector) - 1][0]
    m = int((len(vector) - 1) / (3 * n))
    k = 0
    for i in range(n):
        lst = []
        for j in range(m):
            temp = []
            temp.append(vector[k][0])
            temp.append(vector[k + 1][0])
            temp.append(vector[k + 2][0])
            k += 3
            lst.append(temp)
        image.append(lst)
    return image

def text_from_X_to_T_float(vector):
    text = ''
    for a in vector:
        text += chr(int(a))
    return text

def image_from_X_to_T_float(vector):
    image = []
    n = int(vector[len(vector) - 1][0])
    m = int((len(vector) - 1) / (3 * n))
    k = 0
    for i in range(n):
        lst = []
        for j in range(m):
            temp = []
            temp.append(int(vector[k][0]))
            temp.append(int(vector[k + 1][0]))
            temp.append(int(vector[k + 2][0]))
            k += 3
            lst.append(temp)
        image.append(lst)
    return image
