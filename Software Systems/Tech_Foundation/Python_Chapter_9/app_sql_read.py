# Mehnaz Afrose
import sqlite3
# Must have completed the SQLite writing to Movies database prior to this task
# Read from a SQLite database
print("Getting data")
db_name = "db.sqlite3"
with sqlite3.connect(db_name) as conn:
    sql_command = "SELECT * FROM Movies"
    cursor = conn.execute(sql_command)
    for row in cursor:
        # row is a tuple.
        print(f"The movie {row[1]} was released in {row[2]}")


# Modify the sql_command to filter data
# Remember your SELECT FROM WHERE statements? Now you can use them'
print("Getting data")
db_name = "db.sqlite3"
with sqlite3.connect(db_name) as conn:
    # sql_command can be any select sql statement to get the data you need
    sql_command = "SELECT * FROM Movies WHERE Year > 1985"
    cursor = conn.execute(sql_command)
    for row in cursor:
        # row is a tuple. Access the tuple's values with row.[index_number]
        # each row in the tuple corresponds to a row in the table
        # returned by the sql command
        print(f"The movie {row[1]} was released in {row[2]}")


# Another way to read from a sqlite database using cursor.fetchall()
# which returns all the data into a python list
print("Getting data")
db_name = "db.sqlite3"
with sqlite3.connect(db_name) as conn:
    sql_command = "SELECT * FROM Movies"
    cursor = conn.execute(sql_command)
    movies = cursor.fetchall()
print(movies)
