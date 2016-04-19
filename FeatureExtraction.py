import mmap
import MySQLdb

address = {}
db = MySQLdb.connect(host="localhost",    # your host, usually localhost
                     user="root",         # your username
                     passwd="madu1810",  # your password
                     db="california")        # name of the data base

cur = db.cursor()
cur.execute("SELECT distinct Address FROM california.train ORDER BY Address;")
num = 0
for row in cur.fetchall():
    address[row[0]] = num
    num=num+1

district = {}

cur1 = db.cursor()
cur1.execute("SELECT distinct PdDistrict FROM california.train ORDER BY PdDistrict;")

num = 0
for row in cur1.fetchall():
    district[row[0]] = num
    num=num+1

category = {}

cur2 = db.cursor()
cur2.execute("SELECT DISTINCT category FROM california.train ORDER BY Category;")

num = 0
for row in cur2.fetchall():
    category[row[0]] = num
    num = num+1

db.close()

days = {"Sunday": 1, "Monday": 2, "Tuesday": 3, "Wednesday": 4, "Thursday": 5, "Friday": 6, "Saturday": 7}

with open('train.csv', 'rb') as f:
  m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
  data = m.readline()

  while data:
      data = m.readline()
      print data
      break
