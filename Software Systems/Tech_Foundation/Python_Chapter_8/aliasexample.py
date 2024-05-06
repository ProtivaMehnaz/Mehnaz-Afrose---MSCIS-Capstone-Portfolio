# Mehnaz Afrose
# \# Alias can be used when importing modules. Aliases are shortened names for the module
# You can use the pattern "import module_name as alias_name" 
# An common example is "import numpy as np"  then you just use the prefix "np" in your code
# Another common example is "import pandas as pd" Then just use prefix "pd" in your code
# Don't worry about what numpy and pandas are, just focus on what an alias is. 
# Task: 
# Add a line of code that imports the datetime module as an alias called "dt"
import datetime as dt




# dt is the alias of the imported module "datetime"
# alias are useful to shorten our code. Notice the use of "dt." instead of "datetime."
today = dt.date.today()
print("Today = ", today)
print("Current year:", today.year)
print("Current month:", today.month)
print("Current day:", today.day)

timestamp = dt.date.fromtimestamp(1326244364)
print("Date =", timestamp)

