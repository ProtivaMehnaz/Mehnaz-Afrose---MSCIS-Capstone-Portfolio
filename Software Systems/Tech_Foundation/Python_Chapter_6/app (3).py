# Watching the short videos at codewithmosh.com is very important for
# understanding this chapter
from timeit import timeit

# Chapter 6.1 Exceptions
# nothing to code. Watch the video


# Chapter 6.2 Handling Exceptions
print("\nChapter 6.2 Handling Exceptions")
print("\nTo test the erorrs, enter correct and bad values for age('z', -1, etc.)")
# Run "python app.py" or "python3 app.py" in the terminal to allow for user input
# Start coding here...
# basic error handling
try:
    age = int(input("Age: "))
except ValueError:
    print("You didn't enter, a valid age.")
else:
    print("No exceptions were thrown.")
    print("Execution continues.")
# Pause typing and test your code.
# setting variable as error and printing the error
try:
    age = int(input("Ae: "))
except ValueError as ex:
    print("You didn't enter a valid age.")
    print(ex)
    print(type(ex))
else:
    print("No exceptions were thrown.")
    print("Execution continues.")
# Note: the else clause is not required. Use only if you need it.
# Pause and test your code.


# Chapter 6.3 Handling Different Exceptions
print("\nChapter 6.3 Handling Different Exceptions")
# multiple except clauses to handle multiple errors
# try inputing zero as age to see the message

try:
    age = int(input("Age: "))
    xfactor = 10 / age
except ValueError:
    print("You didn't enter, a valid age.")
except ZeroDivisionError:
    print("Age mu3t be greater, than zero.")
else:
    print("No exceptions were thrown.")
    print(xfactor)
# Pause and test your code.
# one except clause can handle multiple exceptions
try:
    age = int(input("Age: "))
    ifactor = 10 / age
except (ValueError, ZeroDivisionError):
    print("You didn't enter a valid age.")
else:
    print("No exceptions were thrown.")
    print(xfactor)


# Chapter 6.4 Cleaning Up
print("\nchapter 6.4 Cleaning Up")
# finally clause always runs whether there is an error or not
# useful for when opening a file and you need it closed regardless
# if there is an error or not
try:
    file = open("app.py")
    age = int(input("Age: "))
    ifactor = 10 / age
except (ValueError, ZeroDivisionError):
    print("You didn't enter a valid age.")
else:
    print("No exceptions were thrown.")
finally:
    file.close()


# Chapter 6.5 The With Statement
print("\nChapter 6.5 The With Statement")
# State that the with statement can close a file even if there is an error or not
# State that using a with statement removes the need to use a finally: clause
try:
    with open("app.py") as file:
        print("File opened.")
    age = int(input("Age: "))
    xfactor = 10 / age
except (ValueError, ZeroDivisionError):
    print("You didn't enter a valid age.")
else:
    print("No exceptions were thrown.")


# Chapter 6.6 Raising Exceptions print("\nChapter 6.6 Raising Exceptions")


def calculate_xfactor(age):
    if age <= 0:
        raise ValueError("Age cannot be 0 or less.")
    return 10/age


try:
    calculate_xfactor(-1)
    # notice the variable "error".
    # It holds info about the exception. The variable name can be any name you desire.
except ValueError as error:
    print(error)


# Chapter 6.7 Cost of Raising Exceptions
print("\nChapter 6.7 Cost of Raising Exceptions")
# and instead use an if statement
# a best practice is to avoid raising exceptions where possible
code1 = """
def calculate_xfactor(age):
    if age <= 0:
        raise ValueError("Age cannot be 0 or less.") 
    return 10 / age


try:
    calculate_xfactor(-1) 
except ValueError as error: 
    pass
"""


code2 = """
def calculate_xfactor(age): 
    if age <= 0:
        return None
    return 10 / age

xfactor = calculate_xfactor(-1)
if xfactor == None:
    pass
"""
print("first code=", timeit(code1, number=10000))
print("second code=", timeit(code2, number=10000))
