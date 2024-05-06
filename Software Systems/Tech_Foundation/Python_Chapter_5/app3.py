# Mehnaz Afrose
# Import various modules needed for this lesson
from random import random
from array import array
from collections import deque
# start coding below this line

# Chapter 5.13 Stacks
# stacks are lists with LIFO behavior
print("\nChapter 5.13 Stacks" + "-" * 23)
browsing_session = []
browsing_session.append("http://cnn.com")
browsing_session.append("http://wtamu.edu")
browsing_session.append("http://sciencefriday.com")
print(browsing_session)
last = browsing_session.pop()
print(last)
print(browsing_session)
print("redirect to ", browsing_session[-1])
if not browsing_session:
    print("disabled back button")

    # Chapter 5.14 Queues
# Queues are like lists with FIFO behavior
print("\nChapter 5.14 Queues" + "-" * 20)
queue = deque([])
queue.append(1)
queue.append(2)
queue.append(3)
print(queue)
print(f"Now serving customer Cqueue[0]1")
queue.popleft()
print(queue)
print(f"Now serving customer #{queue[0]}")
queue.popleft()
print(queue)
print(f"Now serving customer #{queue[0]}")
queue.popleft()
if not queue:
    print("empty. No more customers.")


# Chapter 5.15 Tuples
print("\nChapter 5.15 Tuples" + "-" * 20)
point = (1, 2, 3)
print(type(point))

point = (1, 2) + (3, 4)
print(point)

point = (1, 2) * 3
print(point)

point = tuple([1, 2])
print(point)

point = (1, 2, 3)
print(point[1])
print(point[0:2])

# unpack a tuple
x, y, z = point
if 10 in point:
    print("exists")

    # Chapter 5.16 Swapping Variables
print("\nChapter 5.16 Swapping Variables" + "-" * 20)
# old way of swapping variable values
x = 10
y = 11
z = x
x = y
y = z
print(x, y)

# bectter way to swap variable values
x = 10
y = 11

x, y = y, x

# assign two variables using this swap trick
a, b = 1, 2

print("x", x)
print('y', y)


# Chapter 5.17 Arrays
print("\nChapter 5.17 Arrays" + "-" * 20)
numbers = array("i", [1, 2, 3])
print(numbers)
numbers.append(4)
print(numbers)
numbers.insert(0, 13)
print(numbers)
numbers.insert(4, 13)
print(numbers)
numbers.pop
print(numbers)
print(numbers[0])
print(numbers[4])
numbers[3] = 99
print(numbers)


# Chapter 5.18 Sets
print("\nChapter 5.18 Sets" + "-" * 20)
numbers = [1, 1, 2, 3, 4]
uniques = set(numbers)
print(uniques)

first = set(numbers)
second = {1, 5}

print(first | second)
print(first & second)
print(first - second)
print(first ^ second)

if 1 in first:
    print("yes")

# Chapter 5.19 Dictionaries
print("\nChapter 5.19 Dictionaries" + "-" * 23)
point = {"x": 1, "y": 2}
point = dict(x=1, y=2)
point["x"] = 10
point["z"] = 20
if "a" in point:
    print(point["a"])
print(point.get("a"))
del point["x"]
print(point)
for key, value in point.items():
    print(key, value)


# Two ways to make dictionaries
nfl_teams = {
    "Dallas": "Cowboys",
    "New England": "Patriots",
    "Green Bay": "Packers",
    "Kansas City": "Chiefs",
    "New Orleans": "Saints",
}
print(nfl_teams)
# Note: When using dict(), the key does not use quotes, the value does.
mlb_teams = dict(
    Boston="Red Sox",
    Houston="Astros",
    Chicago="Cubs",
    Seattle="Mariners",
    Toronto="Blue Jays",
    Dallas="Texas Rangers",
)
print(mlb_teams)
print(mlb_teams["Houston"])
print(mlb_teams["Toronto"])

# add new key value pair
mlb_teams["Miami"] = "Marlines"
print(mlb_teams)

# use .get() to get a key value pair that may not be in in dictionary
print(mlb_teams.get("Anarillo"))

for key, value in mlb_teams.items():
    print(f"City of {key} has team name {value}")


# Chapter 5.20 Dictionary Comprehensions
print("\nChapter 5.20 Dictionary Comprehensions" + "-" * 20)
# populating a dictionary using for loop
values = {}
for x in range(5):
    values[x] = x * 2
# populating a dictionary using dictionary comprehension, similar to list comprehension
values = {x: x * 2 for x in range(5)}
print(values)


# Chapter 5.21 Generator Expressions
print("\nChapter 5.21 Generator Expressions" + "-" * 20)

#	values = (x * 2 for x in range(100000))
#	print("gen:", getsizeof(values))
#	values = [x * 2 for x in range(100000)]
# print(-list:", getsizeof(values))

# Chapter 5.22 Unpacking Operator
print("\nChapter 5.22 Unpacking Operator" + "-" * 20)
first = {"x": 1}
second = {"x": 10, "y": 2}
combined = {**first, **second, "z": 1}
print(combined)

# Chapter 5.23 Exercise
print("\nChapter 5.23 Exercise" + "-" * 20)
sentence = " This is a common interview question"

char_frequency = {}
for char in sentence:
    if char in char_frequency:
        char_frequency[char] += 1
    else:
        char_frequency[char] = 1

char_frequency_sorted = sorted(
    char_frequency.items(),
    key=lambda kv: kv[1],
    reverse=True)

print(char_frequency_sorted[0])
