from pathlib import Path
import json
import sqlite3

"""
Watch Mosh's video 9.8. You will need to install
DB Browser for SQLlite and learn how to create a table
"""


# Chapter 9.8- Working with a SQLite Database
print("\n9.8- Working with a SQLite Database" + "-" * 20)

# Another way to loop through a list of dictionaries. Important for Chapter 9.8 Databases
# a json file is a text files formated similar to a python dictionary
movies = json.loads(Path("movies.json").read_text())
print(movies)
for movie in movies:
    print(movie.values())

 
db_name = "db.sqlite3"
with sqlite3.connect(db_name) as conn:
    #	format INSERT INTO with column names. Use ? marks for value placeholders 
    sql_command = "INSERT INTO Movies (Id, Title, Year) VALUES (?, ?, ?)"
    #	Loop through each row in list
    for movie in movies:
        #   It movies is a list of dictionaries previosly read from a json file
        #	movie is a dictionary
        #	movie, values	returns dictionary values for a row in the list
        #	must convert movie.values() to a tuple on line 53.
        #	A tuple is like a list but cannot be mutated after creation 
        movie_values = tuple(movie.values())
        #	conn.execute() knows to replace the question marks in the
        #	INSERT INTO statement with values from tuple(movies.value()) 
        conn.execute(sql_command, movie_values)
    #	Save the changes to the database. Notice the indentation is outside
    #	the for loop but inside the with block
    conn.commit()
    print("Data written to the database")

