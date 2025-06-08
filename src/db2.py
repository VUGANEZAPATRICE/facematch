from os import name
import sqlite3
# Creating a connection

conn2 = sqlite3.connect("FaceIdentity2.db",check_same_thread=False) #check_same_thread=False
cb = conn2.cursor() # the cursor will be used to execute our sql statement

# Creating a table
def create_tabledb2():
    cb.execute('CREATE TABLE IF NOT EXISTS tasksTabledb2(id INTEGER NOT NULL,name TEXT, address text,age INTEGER, identity_numb INTEGER, task_due_date DATE, photo BLOB)')
    # c.execute('CREATE TABLE IF NOT EXISTS tasksTable(id INTEGER PRIMARY KEY NOT NULL,name TEXT, address text,age INTEGER, identity_numb INTEGER, task_due_date DATE, photo BLOB)')
    # c.execute('CREATE TABLE IF NOT EXISTS tasksTable(id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,name TEXT, identity_numb INTEGER, task_due_date DATE, photo BLOB)')

# function to add task: Create/insert
def add_data(id,name,address,age,identity_numb,task_due_date,photo):
    cb.execute("INSERT INTO tasksTabledb2(id,name,address,age,identity_numb,task_due_date,photo) VALUES (?,?,?,?,?,?,?)",(id,name,address,age,identity_numb,task_due_date,photo))
    conn2.commit() # for saving the above data

def view_all_data():
    cb.execute("SELECT * FROM tasksTabledb2")
    # data = c.fetchall()  
    data = cb.fetchall() 
    return data

def view_unique_tasks():
    cb.execute("SELECT DISTINCT name FROM tasksTabledb2")
    data = cb.fetchall()
    return data

def delete_data(name):
	cb.execute('DELETE FROM tasksTabledb2 WHERE name="{}"'.format(name))
	conn2.commit()

