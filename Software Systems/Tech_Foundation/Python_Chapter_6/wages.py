# calculates paycheck from hourly wages
# Modify this code by adding a try except clause to prevent a "ValueError: could not convert string to float"
# To duplicate the error, enter "z" as a value for either input
# There are two inputs, be sure to fix both in one try except clause so that
# both inputs will prevent errors
# if a ValueError occurs, print the message "Invalid input. Must enter a number."

rate = input("Enter hourly pay rate:")
try:
    rate = float(rate)
    hours_worked = input("Enter hours worked in one week:")
    try:
        hours_worked = float(hours_worked)
        wages = hours_worked * rate
        print(f"Wages: ${wages:.2f}")
    except ValueError:
        print("Invalid input. Must enter a number.")
except ValueError:
    print("Invalid input. Must enter a number.")
