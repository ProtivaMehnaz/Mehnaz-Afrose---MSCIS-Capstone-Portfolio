# Follow the instructions to learn how to create a package of modules
# post.py and likes.py are modules. blog_answer is a package
# View codewithmosh.com Chapter 8 Video 4 for a refresher

import blog.post as post
import blog.likes as likes


# Run "python testblogs.py" to test this code. 
# IF you get an NameError, your import is incorrect
# Code that simulates the posting and likes of a blog website
# Do not modify the code below this line
post.post_blog()
post.post_blog()
post.post_delete()
post.post_edit()

likes.like_blog()
likes.like_blog()
likes.dislike_blog()
likes.leader_board()

