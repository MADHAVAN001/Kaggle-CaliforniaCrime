import mmap
import MySQLdb
import csv

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

k = 0
pre_processed_list = []
with open('train.csv', 'rb') as f:
  m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
  data = m.readline()
  progress = 0
  while data:
      data = m.readline().replace('\n',"").split(',')
      if(not(data)):
          break
      year = int(data[0][0:4])
      month = int(data[0][5:7])
      hour = int(data[0][11:13])
      day = days[data[3]]
      dist = district[data[4]]
      addr = address[data[7]]
      x = float(data[8])
      y = float(data[9])
      desired_output = category[data[1]]
      pre_processed = [year,month,hour,day,dist,addr,x,y]
      pre_processed_list.append(pre_processed)
      k = k+1
      progress +=1
      if(progress%1000 == 0):
          print "Completed: " + str(progress)

with open("train_preprocessed.csv", "wb") as g:
          writer = csv.writer(g)
          writer.writerows(pre_processed_list)