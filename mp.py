print("ENTER WEIGHT MATRIX 1:")
w14 = float(input())
w15 = float(input())
w24 = float(input())
w25 = float(input())
w34 = float(input())
w35 = float(input())

print ("ENTER WEIGHT MATRIX 2 :")

w46 = float(input())
w56 = float(input())

print("ENTER INPUT VALUES :")

x1 = float(input())
x2 = float(input())
x3 = float(input())

x4 = (x1*w14) + (x2*w24)+(x3*w34)
x4 = 1/(1+(2.7**(-x4)))
x5 = (x1*w15)+(x2*w25)+(x3*w35)
x5 = 1/(1+(2.7**(-x5)))

x6 = (x4*w46)+(x5*w56)
x6 = 1/(1+(2.7**(-x6)))

print(x6)

    




