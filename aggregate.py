from multiprocessing.pool import Pool
import glob
import os
import pandas as pd
import math
from pubmed import setup_database
import sqlite3
import traceback

max_retry = 5
NUM_PROCESS = 5
ATTACH_LIMIT = 9 # there is a hard upper bound on SQLITE_MAX_FUNCTION_ARG of 10
MAX_ARGUMENT = 100 # there is a hard upper bound on SQLITE_MAX_FUNCTION_ARG of 127 

def worker(work_id, dbs, dest):
	# merge databases and store their data in the first database
	setup_database(dest)
	conn = sqlite3.connect(dest)
	c = conn.cursor()
	ids = ['db{}'.format(x+work_id) for x in range(len(dbs))]
	
	try:
		for i in range(0, len(dbs), ATTACH_LIMIT):
			for j in range(i, min(len(dbs), i+ATTACH_LIMIT)):
				c.execute("attach '{}' as {}".format(dbs[j], ids[j]))
			c.execute("BEGIN")
			for j in ids[i:min(len(dbs), i+ATTACH_LIMIT)]:
				c.execute("INSERT INTO PubMed SELECT * FROM {}.PubMed".format(j))
				c.execute("INSERT INTO PubMed_Abstract SELECT * FROM {}.PubMed_Abstract".format(j))
			c.execute("COMMIT")
			for j in range(i, min(len(dbs), i+ATTACH_LIMIT)):
				c.execute("detach {}".format(ids[j]))
	except Exception as e:
		print("ERROR in {}: {}".format(dest, e), i, j)
		traceback.print_exc()
		conn.rollback()

	conn.close()

def main(path, dest):
	dbs = sorted(glob.glob(os.path.join(path, '*.db')), key=os.path.getsize)
	# 15341 .db when I ran it.
	# the number here likely disagrees with number of rows in encoding.csv 
	# because I didn't create a .db file for keyword combinations that yielded 
	# no results.
	count = len(dbs)
	counts = pd.read_csv(os.path.join(path, 'encoding.csv'))

	if not os.path.exists(dest):
		os.mkdir(dest)

	# aggregate databases

	# initial merge to limit the number of db files

	BATCH_SIZE = 1000
	count = math.ceil(count/BATCH_SIZE)
	with Pool(processes=NUM_PROCESS) as pool:
		for i in range(count):
			pool.apply_async(worker, (BATCH_SIZE*i, dbs[i*BATCH_SIZE:(i+1)*BATCH_SIZE], os.path.join(dest, '{}.db'.format(i))))
		pool.close()
		pool.join()

def combine(path, dest):
	dbs = sorted(glob.glob(os.path.join(path, '*.db')), key=os.path.getsize)
	worker(0, dbs, dest)

if __name__ == '__main__':
	raw_data_path = 'PubMed'
	aggregate_data_des = 'PubMed_FINAL'
	main(raw_data_path, aggregate_data_des)
	combine(aggregate_data_des, os.path.join(aggregate_data_des, 'pubmed.db'))