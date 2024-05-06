# Mehnaz Afrose
import requests
from bs4 import BeautifulSoup

url = "https://stackoverflow.com/questions"


response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
questions = soup.select(".s-post-summary")

for question in questions:
    question_text = question.select_one(".s-link").getText()
    votes = question.select_one(".s-post-summary--stats-item-number").getText()
    print(question_text, votes)
print("---End Questions on page 1---")


for i in range(2, 5):
    url_suffix = f"?tab=newest&page={i}"
    next_url = url + url_suffix
    print(next_url)
    response = requests.get(next_url)
    soup = BeautifulSoup(response.text, "html.parser")

    for question in questions:
        question_text = question.select_one(".s-link").getText()
        votes = question.select_one(
            ".s-post-summary--stats-item-number").getText()
        print(question_text, votes)
print(f"---End Questions on page {i}---")


"""
If you get an error of
    AttributeError: 'NoneType' object has no attribute 'getText'
the error means you spelled the class selector wrong
in .select() or .select_one()
soup cannot find any class with the wrong name.
Correct your spelling and try again.
"""

"""
Here are the URLs for page 2, 3, and 4. Notice a pattern?
https://stackoverflow.com/questions?tab=newest&page=2
https://stackoverflow.com/questions?tab=newest&page=3
https://stackoverflow.com/questions?tab=newest&page=4

We can build a loop to crawl through more pages.
"""
