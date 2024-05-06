# Mehnaz Afrose
# Simulate blog posts to practice classes and objects
class Blog():
    def __init__(self, message):
        # create two attributes of message and how many likes
        self. message = message
        self.likes = 0

    def post(self):
        # print the post to the terminal
        print(self.message)


# Write several Hog messages and create objects of class Blog()
blogl = Blog("What are the mains points of Chapter 7?")
blog2 = Blog("Classes are blueprints of objects")
blog3 = Blog("Objects are instances of classes")
blog4 = Blog("This is called Object Oriented Programming OOP")
blog5 = Blog("Classes should use Pascal naming convention")

# Output blogs
blogl.post()
blog2.post()
blog3.post()
blog4.post()
blog5.post()

# set some likes on certain blogs
blogl.likes = 2
blog3.likes = 3
print(blogl.likes)
print(blog2.likes)
print(blog3.likes)
