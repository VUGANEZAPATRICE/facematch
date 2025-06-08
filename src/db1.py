from os import name
import sqlite3
# Creating a connection

conn1 = sqlite3.connect("FaceIdentity1.db", check_same_thread=False) 
cc = conn1.cursor() # the cursor will be used to execute our sql statement

# Creating a table
def create_tabledb1():
    cc.execute('CREATE TABLE IF NOT EXISTS tasksTabledb1(id INTEGER NOT NULL,name TEXT, address text,age INTEGER, identity_numb INTEGER, task_due_date DATE, photo BLOB)')
    # c.execute('CREATE TABLE IF NOT EXISTS tasksTable(id INTEGER PRIMARY KEY NOT NULL,name TEXT, address text,age INTEGER, identity_numb INTEGER, task_due_date DATE, photo BLOB)')
    # c.execute('CREATE TABLE IF NOT EXISTS tasksTable(id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,name TEXT, identity_numb INTEGER, task_due_date DATE, photo BLOB)')

# function to add task: Create/insert
def add_data(id,name,address,age,identity_numb,task_due_date,photo):
    cc.execute("INSERT INTO tasksTabledb1(id,name,address,age,identity_numb,task_due_date,photo) VALUES (?,?,?,?,?,?,?)",(id,name,address,age,identity_numb,task_due_date,photo))
    conn1.commit() # for saving the above data

def view_all_data():
    cc.execute("SELECT * FROM tasksTabledb1")
    # data = c.fetchall()  
    data = cc.fetchall() 
    return data

def view_unique_tasks():
    cc.execute("SELECT DISTINCT name FROM tasksTabledb1")
    data = cc.fetchall()
    return data

def delete_data(name):
	cc.execute('DELETE FROM tasksTabledb1 WHERE name="{}"'.format(name))
	conn1.commit()


