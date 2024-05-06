# Create a simple class to represent a person
class Person():
    def __init__(self, first, last):
        self.first_name = first
        self.last_name = last
        self.age = 1

    def set_age(self, age):
        try:
            if 0 <= age <= 130:
                self.age = age
            else:
                print("Age must be a number between 1 and 130")
        except (ValueError, TypeError):
            print("Age must be a number between 1 and 130")

    def speak(self):
        print(f"Hi. My name is {self.first_name} {self.last_name}.")


# create objects of class Person()
# creating an object is call 'instantiation' or instantiate an object' of class Person
p1 = Person("George", "Washington")
p1.set_age(67)
p1.speak()


p2 = Person("John", "Adams")
p2.set_age(90)
p2.speak()


p3 = Person("Thomas", "Jefferson")
p3.set_age(83)
p3.speak()


# test the try except error catching
p4 = Person("Sean", "Humpherys")
p4.set_age("hi there")
print(p4.age)
p4.speak()
