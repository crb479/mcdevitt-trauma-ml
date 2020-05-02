import sqlite3
import os

folder = 'PubMed_FINAL'
raw_path = os.path.join(folder, 'raw_pubmed.db')
original_path = os.path.join(folder, 'pubmed.db') # the original .db is renamed "raw_pubmed." pubmed.db now only contains the PubMed table.
new_path = os.path.join(folder, 'clean_pubmed.db')

def extract_pubmed():
	conn = sqlite3.connect(original_path)
	c = conn.cursor()
	# setup
	c.execute("""
		CREATE TABLE IF NOT EXISTS PubMed(
			TITLE varchar(512) NOT NULL,
			K1 varchar(128) NOT NULL,
			K2 varchar(128) NOT NULL,
			DOI varchar(256),
			AUTHORS varchar(500),
			TYPE varchar(20),
			PUBLICATIONYEAR var(4),
			URL varchar(512)
		)
	""")

	c.execute("attach '{}' as db".format(raw_path))
	c.execute('BEGIN')
	c.execute("INSERT INTO PubMed SELECT * FROM db.PubMed")
	c.execute('COMMIT')
	c.execute("detach db")

def clean_pubmed():
	conn = sqlite3.connect(new_path)
	c = conn.cursor()

	# setup
	c.execute("""
		CREATE TABLE IF NOT EXISTS PubMed(
			TITLE varchar(512) NOT NULL,
			K1 varchar(128) NOT NULL,
			K2 varchar(128) NOT NULL,
			DOI varchar(256),
			AUTHORS varchar(500),
			TYPE varchar(20),
			PUBLICATIONYEAR var(4),
			URL varchar(512)
		)
	""")

	c.execute("attach '{}' as db".format(original_path))
	c.execute('BEGIN')
	c.execute("""
		INSERT INTO PubMed(TITLE, K1, K2, DOI, AUTHORS, TYPE, PUBLICATIONYEAR, URL)
			SELECT TITLE, K1, K2, DOI, AUTHORS, TYPE, PUBLICATIONYEAR, URL FROM db.PubMed
			WHERE (
				db.PubMed.TITLE IN (
					SELECT TITLE FROM db.PubMed 
					GROUP BY TITLE
					HAVING COUNT(COALESCE(DOI, '0')) > 1 
				)
				AND db.PubMed.DOI IS NOT NULL
			)
			OR (
				db.PubMed.TITLE IN (
					SELECT TITLE FROM db.PubMed 
					GROUP BY TITLE
					HAVING COUNT(COALESCE(DOI, '0')) == 1 
				)
			)
	""")
	c.execute('COMMIT')
	c.execute("detach db")

'''
Rules for cleaning:
1. remove entries with the same title. if two entries have the same title, prioritize the one with DOI. if both lack DOI, pick one from them.
'''

# pre cleaning: 15662230 articles
# post cleaning: 12505293 articles