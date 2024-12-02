import turtle

# Create a Turtle screen
screen = turtle.Screen()
screen.bgcolor("green")

# Create a Turtle object
pen = turtle.Turtle()
pen.speed(1)

# Draw the letter "P"
pen.penup()
pen.setpos(-100, 0)
pen.pendown()
pen.circle(60, 180)
pen.left(90)
pen.forward(180)
pen.penup()

# Export the path to a file
# with open("letter_p.txt", "w") as f:
#     f.write(pen.get_path())

# Close the Turtle graphics window
screen.exitonclick()