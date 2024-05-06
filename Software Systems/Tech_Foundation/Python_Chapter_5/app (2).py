# Mehnaz Afrose
# Import various modules needed for this lesson
from random import random
from array import array
from collections import deque
# start coding below this line

# Chapter 5.1 Lists
print("Chapter 5.1 Lists" + "-" * 20)
letters = ["a", "b", "c"]
matrix = [[0, 1], [2, 3]]
zeros = [0] * 5
combined = zeros + letters
numbers = list(range(20))
chars = list("Hello World")
print(len(chars))

women_in_computing = ["Ada Lovelace", "Grace Hopper", "Sister Mary Keller"]
star_wars_release = [1977, 1980, 1983, 1999, 2020, 2005, 2015, 2017, 2019]
print(len(women_in_computing))
print(len(star_wars_release))


# Chapter 5.2 Accessing Items
print("\nChapter 5.2 Accessing Items" + "-" * 20)
letters = ["a", "b", "c", "d"]
letters[0] = "A"
print(letters)
print(letters[0:3])
print(letters[:3])
print(letters[1:])

numbers = list(range(20))
print(numbers[::-1])
print(numbers[::2])

# Change values in a list
print(women_in_computing)
women_in_computing[1] = "race Brewster Murray Hopper"
women_in_computing[2] = "Sister Mary Kenneth Keller"
print(women_in_computing)

# Return specific items in a list
print(women_in_computing[0])
print(women_in_computing[-1])
print(star_wars_release[3:6])

# variables can be used as index numbers
i = 7
j = 8
print(star_wars_release[i])
print(star_wars_release[j])


# 4 Chapter 5.3 List Unpacking
print("\nChapter 5.3 List Unpacking"+"-" * 20)
numbers = [1, 2, 3]
first, second, third = numbers
print(first, second, third)

numbers = [1, 2, 3, 4, 4, 4, 4, 4]
first, second, *other = numbers
print(first)
print(*other)

numbers = [1, 2, 3, 4, 4, 4, 4, 9]
first, second, *other = numbers
print(first, second)
print(other)

numbers = [1, 2, 3, 4, 4, 4, 4, 9]
first, *other, last = numbers
print(first, last)
print(other)


def multiply(*numbers):
    total = 1
    for x in numbers:
        total *= x
    return total


answer = multiply(5, 8, 55, 49, 65, .05, 100)
print(answer)

# List inside a list. Sales in millions
star_wars_box_office = [
    ["Star Wars", 1300],
    ["The Empire Strikes Back", 704.2],
    ["Return of Jedi", 723.2],
    ["The Phantom Menace", 757.5]
]

print(star_wars_box_office[0][0])
print(star_wars_box_office[1][1])
print(star_wars_box_office[1][0])
print(star_wars_box_office[1][1])
print(star_wars_box_office[2][0])
print(star_wars_box_office[2][1])
print(star_wars_box_office[3][0])
print(star_wars_box_office[3][1])

# Chapter 5.4 Looping over Lists
print("\nChapter 5.4 Looping over Lists" + "-" * 20)
letters = ["a", "b", "c"]
for letter in letters:
    print(letter)

items = [0, "a"]
for index, letter in enumerate(letters):
    print(index, letter)

for lady in women_in_computing:
    print(lady)

for year in star_wars_release:
    print(year)
# Note. lady and year are just variable names. You can use any variable name.
# Best practice is to name lists with plural nouns and
# use the singular noun in the for loop

# Looping through a list inside a list using unpacking
for movie, total in star_wars_box_office:
    print(f"Box 'Office for {movie} were $ {total} million")

# Looping through a list of tuples using unpacking
sales_per_month = [("Clan", 5000), ("Feb", 4050), ("Mar", 3752)]
for month, total in sales_per_month:
    print(f"Sales for {month} were $ {total}")


# Chapter 5.5 Adding or Removing Items
print("\nChapter 5.5 Adding or Removing Items" + "-" * 20)
letters = ["a", "b", "c"]

# Add
letters.append("d")
letters.insert(0, "-")


# Remove
letters.pop()
letters.remove("b")
del letters[0:3]
letters.clear()
print(letters)
# practice
vegies = ["onion", "green beans", "corn", "carrot"]
more_vegies = ["zucchini", "potato", "beat"]
vegies = vegies + more_vegies
print(vegies)
vegies.append("radish")
print(vegies)
vegies.insert(2, "lima bean")
print(vegies)
print(vegies)
vegies.remove("corn")
print(vegies)
# insert a list inside a list at index 2
still_more_vegies = ["cabbage", "garlic", "bok choy"]
vegies[2: 2] = still_more_vegies
print(vegies)


# Chapter 5.6 Finding Items
print("\nChapter 5.6 Finding Items" + "-" * 20)
letters = ["a", "b", "c"]
print(letters.index("a"))

if "d" in letters:
    print(letters.index("d"))

print(letters.count("d"))

i = vegies.index("garlic")
print(vegies[i])

if "carrot" in vegies:
    index = vegies.index("carrot")
    print(index)
    vegies.pop(index)
print(vegies)


# Chapter 5.7 Sorting Lists
print("\nChapter 5.7 Sorting Lists" + "-" * 20)
numbers = [3, 51, 2, 8, 6]
numbers.sort()
print(numbers)
numbers.sort(reverse=True)
print(numbers)

new_list_sorted = sorted(numbers)
print(new_list_sorted)

# sort in descending order
new_list_sorted = sorted(numbers, reverse=True)
print(new_list_sorted)

# What if you want to sort a complex object, such as items, by quanity
items = [
    ("Product1", 10),
    ("Product2", 9),
    ("Product4", 7),
    ("Product5", 12)
]

# does not sort by quanity, only product name
items.sort()
print(items)

# sort by quanity.


def sort_item(item):
    # sort by quantity, which is index 1 in the list of tuples
    return item[1]


# pass the function name to .sort()
items.sort(key=sort_item)
print(items)

# another example
print(star_wars_box_office)
star_wars_box_office.sort(key=sort_item)
print(star_wars_box_office)
