import mmap
import MySQLdb
import csv
import sys
from csv import reader

db = MySQLdb.connect(host="localhost",    # your host, usually localhost
                     user="root",         # your username
                     passwd="madu1810",  # your password
                     db="california")        # name of the data base

category = {}

cur2 = db.cursor()
cur2.execute("SELECT DISTINCT category FROM california.train ORDER BY Category;")

num = 0
for row in cur2.fetchall():
    category[row[0]] = num
    num = num+1

db.close()

pre_processed_list = []
with open('train.csv', 'rb') as f:
  progress = 0
  for data in reader(f):
      if(progress == 0):
          progress+=1
          continue

      desired_output = category[data[1]]
      pre_processed = [desired_output]
      pre_processed_list.append(pre_processed)

      progress +=1
      if(progress%2000==0):
          print "Completed: " + str(progress)
          sys.stdout.flush()
          with open("train_labels.csv", "a") as g:
              writer = csv.writer(g, delimiter=',', lineterminator='\n')
              writer.writerows(pre_processed_list)
              pre_processed_list = []
              g.close()
