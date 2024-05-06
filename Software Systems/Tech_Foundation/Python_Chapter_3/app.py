# Mehnaz Afrose
# Chapter 3.1 Introduction, no code
# Chapter 3.2 Conditional Statements
# print("Chapter 3.2 Conditional Statements")
temperature = 15
if temperature > 30:
    print("It's warm")
    print("Drink water")
elif temperature > 20:
    print("It's nice")
else:
    print("It's cold")
    print("Done")
print("*" * 30)
# Chapter 3.3 Ternary Operator print("Chapter 3.3 Ternary Operator")
age = 12
message = "Eligible" if age >= 18 else "Not eligible"
print(message)


cost = 300
status = "expensive" if cost > 100 else "cheap"
print(status)
print("*"*30)

# chapter 3.4 Logical Operators
print("##chapter 3.4 Logical Operators")
high_income = False
good_credit = True
student = False

if (high_income or good_credit) and not student:
    print("Eligible")
else:
    print("Not eligible")
print("*" * 30)

# Chapter 3.5 Short-circuit Evaluation
print("#Chapter 3.5 Short-circuit Evaluation")
high_income = False
good_credit = True
student = True
if high_income or good_credit or not student:
    print("Eligible")

# Chapter 3.6 chaining Comparison Operators
print("#Chapter 3.6 Chaining Comparison Operators")
# age should be between 18 and 65
age = 22
if 18 <= age < 65:
    print("Eligible")
print("T" * 30)


# Chapter 3.9 For Loops print("#Chapter 3.8 For Loops")
for number in range(3):
    print("Attempt", number + 1, (number + 1) * ".")
for number in range(1, 4):
    print("Attempt", number + 1, (number + 1) * ".")
# How to count by twos
for number in range(1, 10, 2):
    print("Attempt", number + 1, (number + 1) * ".")
# miow to count down from 10 to zero
for number in range(10, 0, -1):
    print(number)
print("*" * 30)
# Chapter 3.9 For—Else
print("#Chapter 3.9 For..Else")
# it failed
successful = False
for number in range(3):
    print("Attempt")
    if successful:
        print("Successful")
        break
else:
    print("Attempted 3 times and failed")
print("*" * 30)


# if successful
successful = True
for number in range(3):
    print("Attempt")
    if successful:
        print("Successful")
        break
else:
    print("Attempted 3 times and failed")
print("*" * 30)
# Chapter 3.10 Nested Loops
print("#Chapter 3.10 Nested Loops")
for x in range(5):
    for y in range(3):
        print(f"({x}, {y})")
# chapter 3.11 Iterables
print("*Chapter 3.11 Iterables")
for x in (1, 2, 3, 4):
    print(x)
print("*" * 30)


# Chapter 3.12 While Loops
print("#Chapter 3.12 While Loops")
command = ""
while command.lower() != "quit":
    command = input("Enter any command or 'quit' >")
    print("EcHo", command)
print("*" * 30)
# Chapter 3.13 infinite Loops
print("#Chapter 3.13 infinite Loops")
while True:
    command = input("Enter any command or 'quit' >")
    print("ECHO", command)
    if command.lower() == "quit":
        break
print("*" * 30)
# Chapter 3.14 Exercise
print("#Chapter 3.14 Exercise")
counter = 0
for number in range(1, 10):
    if number % 2 == 0:
        print(number)
        counter += 1
print(f"we have {counter} even numbers")
print("*" * 30)
