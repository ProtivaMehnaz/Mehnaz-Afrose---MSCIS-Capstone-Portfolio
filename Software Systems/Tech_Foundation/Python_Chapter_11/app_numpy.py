# Mehnaz Afrose
import numpy as np

# Convert list to an Numpy array
simple_list = [1, 2, 3]
array = np.array(simple_list)
print(array)
print(type(array))

# Convert multidimensional list to a Numpy array
print("	multidemensional array")
multidimensional_list = [
    [1, 2, 3],
    [4, 5, 6]
]
array = np.array(multidimensional_list)
print(array)
print(type(array))
print(f"Dimensions are {array.shape}")

# Initializing arrays with default values
# specify the dimensions and data type
print("-----Initializing arrays")
array = np.zeros((2, 4), dtype=int)
print(array)
array = np.ones((2, 4), dtype=int)
print(array)
array = np.full((2, 4), 99, dtype=int)
print(array)

array = np.random.random((2, 4))
print(array)

# Accessing elements in the array
print("-----Accessing elements in the array")
print(array[0,	0])
print(array[0,	1])
print(array[1,	0])
print(array[1,	1])

# loop through every element in the array
print("-----Loop through elements in array")
for element in array:
    print(element)

# loop through every element in the array
print("-----Loop through individual items in array")
for element in array:
    for item in element:
        print(item)

print("-" * 30)
# comparison operators on numpy
print(array > 0.2)

# Filtering array values using comparison operators
print(array[array > 0.2])

# Math operations on numpy
print(np.round(array))

print("-" * 30)
# How to perform math operations on an array, easily.

first = np.array([1, 2, 3])
second = np.array([1, 2, 3])
print(first + second)
print(first - second)
print(first * second)
print(first + 100)

print("----convert inches to cm")
dimensions_inch = np.array([1, 2, 3])
dimensions_cm = dimensions_inch * 2.54
print(dimensions_cm)

# How to convert an array using list comprehension rather than numpy
print("----convert inches to cm")
dimension_cm = [x * 2.54 for x in dimensions_inch]
print(dimensions_cm)
