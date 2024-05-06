# Mehnaz Afrose
# Code a class that describe planes following the instructions.
# We are just simulating a simple plane. You can imagine
# these methods could control a plane's functionality
# We are just simulating a simple plane. You can imagine
# these methods could control a plane's -Functionality
class Plane():
    # create class attributes with default values
    empty_weight = 1043
    wing_span = 13.1
    engines = 1
    make = "Cessna 350"
    model = "LC40-550FG"
    speed = 0

    def __init__(self, given_name):
        # every plan needs a name
        self.name = given_name

    def throttle_engine(self, amount):
        # changes the engine throttle
        self.speed += amount
        print(f"Speed is now {self.speed} knots")

    def pitch(self, degrees):
        # changes altitude
        if degrees > 0:
            print("Up, up, and away")
        elif degrees < 0:
            print("Going down")
        else:
            print("Level flight")

    def roll(self, degrees):
        # banks the plane left and right
        if degrees > 0:
            print("Turning right")
        elif degrees < 0:
            print("Turning left")
        else:
            print("Straight flight")


# Create two plane objects of class Plane()
plane1 = Plane("Flying Blue Monkey")
print(plane1.name)
plane1.throttle_engine(10)
plane1.throttle_engine(50)
plane1.throttle_engine(90)
plane1.pitch(0.10)
plane1.pitch(0)
plane1.throttle_engine(-20)
plane1.roll(.03)
plane1.roll(-.03)
plane1.roll(0)


print("\n\n")
plane2 = Plane("Spirit of Saint Louis")
plane2.empty_weight = 975
plane2.wing_span = 14
plane2.engines = 1
plane2.make = "customized Wright -5 Whirlwind"
plane2.model = "Ryan Airlines custom built"
print(plane2.name)
print("Start in Long Island, New York")
plane2.throttle_engine(10)
plane2.throttle_engine(50)
plane2.throttle_engine(100)
plane2.pitch(0.15)
plane2.pitch(0)
plane2.throttle_engine(-20)
plane2.roll(-.01)
plane2.roll(.01)
plane2.roll(0)
plane2.pitch(-.15)
plane2.throttle_engine(-140)
print(f"{plane2.name} landed in Paris, France. First solo nonstop transatlantic flight!")
