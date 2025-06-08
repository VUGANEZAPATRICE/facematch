from os import name
import sqlite3
# Creating a connection

conn3 = sqlite3.connect("FaceIdentity3.db", check_same_thread=False) 
ca = conn3.cursor() # the cursor will be used to execute our sql statement

# Creating a table
def create_table():
    ca.execute('CREATE TABLE IF NOT EXISTS tasksTable(id INTEGER NOT NULL,name TEXT, address text,age INTEGER, identity_numb INTEGER, task_due_date DATE, photo BLOB)')
    # c.execute('CREATE TABLE IF NOT EXISTS tasksTable(id INTEGER PRIMARY KEY NOT NULL,name TEXT, address text,age INTEGER, identity_numb INTEGER, task_due_date DATE, photo BLOB)')
    # c.execute('CREATE TABLE IF NOT EXISTS tasksTable(id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,name TEXT, identity_numb INTEGER, task_due_date DATE, photo BLOB)')

# function to add task: Create/insert
def add_data(id,name,address,age,identity_numb,task_due_date,photo):
    ca.execute("INSERT INTO tasksTable(id,name,address,age,identity_numb,task_due_date,photo) VALUES (?,?,?,?,?,?,?)",(id,name,address,age,identity_numb,task_due_date,photo))
    conn3.commit() # for saving the above data

def view_all_data():
    ca.execute("SELECT * FROM tasksTable")
    # data = c.fetchall()  
    data = ca.fetchall() 
    return data

def view_unique_tasks():
    ca.execute("SELECT DISTINCT name FROM tasksTable")
    data = ca.fetchall()
    return data

def delete_data(name):
	ca.execute('DELETE FROM tasksTable WHERE name="{}"'.format(name))
	conn3.commit()

