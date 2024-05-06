# Mehnaz Afrose
# This file lets you practice importing functions from modules
# random and math are predefined python module with many functions, 
# ecommerce-related functions come from the ecommerce.py file.
# But all these must be imported into your code before use
# Task: 
# Add a line of code that imports the choice and randint functions from the random module
# Add a line of code that imports the pow, sqrt, floor functions from the math module
# Use the pattern "from x import y, z"
# View codewithmosh.com Chapter 8 Video 1 for a refresher
from random import choice, randint
from math import pow, sqrt, floor
from ecommerce import calc_tax, calc_shipping






# Don't change the code below. The code will work if you get the import functions correct.
# This code will give a "NameError" error if the import function is not correct
# Review this code visually so you understand how to use several useful functions
customers =  ["John", "Mary", "Martha", "Juan", "Liang"]
winning_customer = choice(customers)
print(winning_customer)


# Test functions from the ecommerce module
print(calc_tax(100))
print(calc_shipping("TX"))

# generate five randome numbers between 10 and 100 (not including 100)
for i in range(5):
    x = randint(10, 100)
    print(x)

# Raise to the power
x = 45
y = pow(x, 2)
print(y)

# Square root
print(sqrt(x))

# Round down using floor()
print(f"Floor function rounds down: {floor(132.89794)} ")





