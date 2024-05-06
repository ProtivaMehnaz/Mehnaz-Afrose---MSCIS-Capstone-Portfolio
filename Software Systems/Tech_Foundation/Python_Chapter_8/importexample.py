# Mehnaz Afrose
#  This file lets you practice importing functions from modules

# Task:
# Add a line of code that imports the sys module
# Add a line of code that imports the ecommerce module
# Use the pattern "import module_name"
# View codewithmosh.com Chapter 8 Video 3 for a refresher
import sys
import ecommerce


# Don't change the code below. The code will work if you get the import functions correct.
# This code will give a "NameError" error if the import function is not correct
# Review this code visually so you understand how to use several useful functions
# Notice when using the "import sys" we must use sys as a prefix to sys.path.
# Notice when using the "import ecommerce" we must use ecommerce as a prefix to ecommerce.calc_tax().
print(sys.path)

tax = ecommerce.calc_tax(120)
shipping = ecommerce.calc_shipping("FL")
print(f"Tax: {tax}")
print(f"Shipping: {shipping}")
