# Mehnaz Afrose
import webbrowser

# Chapter 9.7- Opening the Browser
# Put your code here

import webbrowser

print("Deployment Completed")
webbrowser.open("http://google.com")
webbrowser.open("http://wtamu.edu")


# Example:
# Quickly setup your coding environment
# remember that r"" means raw string so python wont be confused by back slashes
learning_objectives = r"https://docs.google.com/document/d/129OHqS-IDEwtqFlamf7S6wHl6pt6VqKZu2CHqCzCDqk/edit#"
code_with_mosh = r"https://codewithmosh.com/courses/417695/lectures/8417582"
assignment = r"https://docs.google.com/document/d/1t8Vg0uw0ZvCzOGpTyU4fQj2MtulxFpjaT1jCnqJ3KkY/edit"
github = r"https://github.com/"
python_help = r"https://www.w3schools.com/python/python_reference.asp"
many_urls = [learning_objectives,
             code_with_mosh,
             github,
             python_help,
             assignment]

for url in many_urls:
    webbrowser.open(url)




# Want to create a batch file that runs python code, visit
# https://automatetheboringstuff.com/2e/appendixb/
