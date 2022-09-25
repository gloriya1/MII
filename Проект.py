import turtle
t = turtle.Turtle()
t.shape('turtle')
t.left(90)
n = 50

def butterfly(n):
    t.circle(n)
    t.circle(-n)

x = 1
while x <= 20:
    butterfly(n)
    n += 5
    x += 1
turtle.exitonclick()