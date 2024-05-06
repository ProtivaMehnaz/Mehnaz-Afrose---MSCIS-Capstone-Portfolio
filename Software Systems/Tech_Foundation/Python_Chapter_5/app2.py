# Mehnaz Afrose
# Import various modules needed for this lesson
from random import random
from array import array
from collections import deque
# start coding below this line

# Chapter 5.8 Lambda Functions
print("\nChapter 5.8 Lambda Functions" + "-" * 20)
items = [
    ("Product2",	10),
    ("Product2",	9),
    ("Product3",	12),
]

# sort by quantity
items.sort(key=lambda item: item[1])
print(items)

# sort by sales
star_wars_box_office = [
    ["Star Wars", 1300],
    ["The Empire Strikes Back", 704.2],
    ["Return of Jedi", 723.2],
    ["The Phantom Menace", 757.5]
]
star_wars_box_office.sort(key=lambda item: item[1])
print(star_wars_box_office)

# sort by movie name
star_wars_box_office.sort(key=lambda item: item[0])
print(star_wars_box_office)


# Chapter 5.9 Map Function
print("\nChapter 5.9 Map Function" + "-" * 20)
# Mapping means to create a new list from another, or transform the list
items = [
    ("Productl",	10),
    ("Product2",	9),
    ("Product3",	12),
]

# transform list into a new list, using for loop
prices = []
for item in items:
    prices.append(item[1])

# transform list into a new list, using lambda
items = [
    ("Productl",	10),
    ("Product2",	9),
    ("Product3",	12),
]

prices = list(map(lambda item: item[1], items))
print(prices)


# Chapter 5.10 Filter Function
print("\nChapter 5.10 Filter Function" + "-" * 20)
items = [
    ("Productl",	10),
    ("Product2",	9),
    ("Product3",	12),
]

filtered = list(filter(lambda item: item[1] >= 10, items))
print(filtered)

# anotner example
sw = [
    ["Star Wars", 1300],
    ["The Empire Strikes Back", 704.2],
    ["Return of ]edi", 723.2],
    ["The Phantom Menace", 757.5]
]
sw_filtered = list(filter(lambda item: item[1] >= 750.0, sw))
print(sw_filtered)


# Chapter 5.11 List Comprehensions
print("\n Chapter 5.11 List Comprehensions" + "-" * 20)
# Mapping means to create a new list from another, or' transform the list
# This is also called "List Comprehension"
items = [
    ("Productl", 10),
    ("Product2", 9),
    ("Product3", 12),
]

# mapping using lambda
prices = list(map(lambda item: item[1], items))

# mapping using pytnori s iisT comprehension. wnicn co you llKef
prices = [item[1] for item in items]
print(prices)

# real example of list comprehension
# create 10 random numbers
random_numbers = [random() for item in range(10)]
print(random_numbers)

# filter using lambda
filtered = list(filter(lambda item: item[1] >= 10, items))

# filter using python's list comprehension. Which do you like?
filtered = [item for item in items if item[1] >= 10]
print(filtered)

# another example using Star Wars list of box office sales
sw_filtered = list(filter(lambda item: item[1] >= 750.0, sw))
sw_filtered = [item for item in sw if item[1] >= 750.0]
print(sw_filtered)

#
# Chapter 5.12 Zip Function
print("\nChapter 5.12 Zip Functio" + "-" * 20)
# Zipping means combining two or more lists into a list of tuple pairs
listl = [1, 2, 3]
list2 = [10, 20, 30]

# we want a list of tuples pairs like this [(1,10), (2, 20), (3, 10)]
list_combined = list(zip(listl, list2))
print()

# Example
products = ["productl", "product2", "product3", "product4", "product5"]
sales = [435, 665, 13, 2425, 596]
quantity_sold = [5, 19, 1, 30, 15]
product_sales_quantity_combined = list(zip(products, sales, quantity_sold))
print(product_sales_quantity_combined)
# extra practice with for loop
# Note: product, sale, and quantity are any variable names you desire
for product, sale, quantity in product_sales_quantity_combined:
    print(f"Sales for {product} were $ {sale} with {quantity} units sold.")
