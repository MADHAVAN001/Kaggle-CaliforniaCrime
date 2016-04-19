import mmap
import MySQLdb

Address = {}
db = MySQLdb.connect(host="localhost",    # your host, usually localhost
                     user="root",         # your username
                     passwd="madu1810",  # your password
                     db="california")        # name of the data base

cur = db.cursor()
cur.execute("SELECT distinct Address FROM california.train ORDER BY Address;")
num = 0
for row in cur.fetchall():
    Address[row[0]] = num
    num=num+1

print Address

db.close()
with open('train.csv', 'rb') as f:
  m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
  data = m.readline()
  days = {"Sunday":1, "Monday":2, "Tuesday":3,"Wednesday":4,"Thursday":5,"Friday":6,"Saturday":7}
  while data:
      data = m.readline()
      print data
      break
