# Mehnaz Afrose
# Create a Robot class according to the instructions
# import random will have functions useful for this assignment
import random

# Start coding here...


class Robot():

    def __init__(self, name):
        self.name = name

    def hello(self):
        print(
            f'Hi. My name is {self.name}. I can count and sing. And, I have feelings')

    def sing(self):
        print('Mary had a little lamb…')

    def count(self, stop_number):
        for i in range(stop_number):
            print(i)

    def how_do_you_feel(self):
        possible_feelings = [":-)", ":-O", ":-(", ":-P"]
        current_feeling = random.choice(possible_feelings)
        print(f'I feel {current_feeling}')


# Test code. DO NOT CHANGE THIS TEST CODE
try:
    r = Robot("Marvin")
    r.hello()
    r.sing()
    r.count(10)
    r.how_do_you_feel()
except AttributeError as err:
    print("***Test failed***")
    print("Missing a method. Making progress but not done yet. Keep coding and try again.")
    print(err)
else:
    print("Test successful. Your code passes")
