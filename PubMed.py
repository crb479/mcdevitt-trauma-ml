import requests
import urllib.request
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from Bio import Entrez
import traceback
import sqlite3

max_retry = 1
date_format = ['Year', 'Month', 'Day']

# create database
def setup_database():
	conn = None
	try:
		conn = sqlite3.connect('./pubmed.db')
		# print(sqlite3.version)
	except Error as e:
		print(e)
	finally:
		if conn:
			c = conn.cursor()
			c.execute("PRAGMA foreign_keys = ON")
			c.execute('''
				CREATE TABLE IF NOT EXISTS PubMed(
					TITLE varchar(512) NOT NULL,
					K1 varchar(128) NOT NULL,
					K2 varchar(128) NOT NULL,
					DOI varchar(256),
					AUTHORS varchar(500),
					TYPE varchar(20),
					PUBLICATIONYEAR var(4),
					URL varchar(512),
					PRIMARY KEY (TITLE)
				)
			''')
			c.execute('''
        		CREATE TABLE IF NOT EXISTS PubMed_Abstract(
        			TITLE varchar(512) NOT NULL REFERENCES PubMed(TITLE) ON DELETE CASCADE,
        			K1 varchar(128),
        			K2 varchar(128),
        			ABSTRACT text,
					PRIMARY KEY (TITLE)
        		)
			''')
			conn.commit()
			conn.close()

# extract and index keywords
def keyword(filepath='./Keywords for Systematic Review Analysis for Trauma by DV-01-29-2020.docx'):
    import docx
    f = docx.Document(filepath)
    first_keyword, second_keyword, exclude = [], [], []
    container, i, j, count = None, [0], [0], None

    # extract keywords and divide them based on which set they belong to
    for t in f.paragraphs[1:]:
    	if len(t.text.strip()) == 0:
    		continue
    	elif 'FIRST KEYWORD' in t.text:
    		container = first_keyword
    		count = i
    	elif 'SECOND KEYWORD' in t.text:
    		container = second_keyword
    		count = j
    	elif 'Exclude' in t.text:
    		exclude = t.text[t.text.index(':')+2:].split(', ')
    		break
    	else:
    		container.append(t.text.strip().lower())
    
    return first_keyword, second_keyword, exclude

# worker function for multithreading
def worker(first, second):
	count = 0
	url = 'https://www.ncbi.nlm.nih.gov/pubmed?term=({}) AND {}'.format(first, second)

	try:
		response = requests.get(url)
		# check connection
		if response.status_code != 200:
			raise Exception

		soup = BeautifulSoup(response.text, 'html.parser')

		# get number of articles in total
		count = soup.findAll("h3", {"class": "result_count left"})
		# a scenario is PubMed directly sending us to the abstract
		if len(count) == 0:
			count = 1
		else:
			count = int(count[0].get_text().split(' ')[-1])

		for _ in range(max_retry):
			try:
				handle = Entrez.esearch(db='pubmed', 
			                            retmax='{}'.format(count),
			                            retmode='xml', 
			                            term='({}) AND ({})'.format(first, second))
				results = Entrez.read(handle)
				id_list = results['IdList']
				# fetch the details for all the retrieved articles
				if len(id_list) > 0:
					handle = Entrez.efetch(db='pubmed',
				                           retmode='xml',
				                           id=id_list)
					r = Entrez.read(handle)['PubmedArticle']
					if r:
						# connect to database
						conn = sqlite3.connect('./pubmed.db')
						c = conn.cursor()
						remaining = 0
						while remaining < count:
							handle = Entrez.efetch(db='pubmed',
						                           retmode='xml',
						                           id=','.join(id_list[remaining:]))
							results = Entrez.read(handle)
							for r in results['PubmedArticle']:
								# collect information
								title = str(r['MedlineCitation']['Article']['ArticleTitle'])
								# print('<title>', title)
								abstract = ' '.join([str(x) for x in r['MedlineCitation']['Article']['Abstract']['AbstractText']]) if 'Abstract' in r['MedlineCitation']['Article'] else None
								# print('<abstract>', abstract)
								doi = None
								for x in r['PubmedData']['ArticleIdList']:
									if x.attributes['IdType'] == 'doi':
										doi = x
								# print('<doi>', doi)
								url = 'http://sci-hub.tw/{}'.format(doi) if doi else None
								authors = ['{} {}'.format(x['LastName'], x['ForeName']) for x in r['MedlineCitation']['Article']['AuthorList']] if 'AuthorList' in r['MedlineCitation']['Article'] else None # [last name] [first name]
								# print('<authors>', authors)
								article_type = str(r['MedlineCitation']['Article']['PublicationTypeList'][0])
								# print('<article_type:', article_type)
								publication_year = r['PubmedData']['History'][0]['Year']
								# print('<publication_year>', publication_year)
								# test url
								if url:
									response = requests.head(url)
									if not response.ok: 
										url = ''
								# insert into database
								c.execute('INSERT OR IGNORE INTO PubMed(TITLE, K1, K2, DOI, AUTHORS, TYPE, PUBLICATIONYEAR, URL) VALUES (?, ?, ?, ?, ?, ?, ?, ?)', 
									(title, first, second, doi, ','.join(authors) if authors else authors, article_type, publication_year, url))
								c.execute('INSERT OR IGNORE INTO PubMed_Abstract(TITLE, K1, K2, ABSTRACT) VALUES (?, ?, ?, ?)', 
									(title, first, second, abstract))
								conn.commit()
								# print()
							remaining += len(results['PubmedArticle'])
						conn.close()
					else:
						count = 0
				else:
					count = 0
				print('SUCCESS: "{}, {}" -> {}\n'.format(first, second, count), end='')
				return count
			except Exception as e:
				print('WARNING: Retrying keywords "{}" and "{}": {}\n'.format(first, second, e), end='')
				traceback.print_exc()
				continue
		print('ERROR: "{}" and "{}"\n'.format(first, second), end='')
	except Exception as e:
		print('ERROR: "{}" and "{}": {}\n'.format(first, second, e), end='')
		traceback.print_exc()

	return count

# search PubMed for paper and the count and search results to 
def search(first_keyword, second_keyword, exclude):
	# multithreading
	for k1 in first_keyword:
		for k2 in second_keyword:
			if '{} {}'.format(k1, k2) not in exclude:
				executor.submit(worker, k1, k2)

# write search counts to csv
def write_result_count(first_keyword, second_keyword):
	import csv
	conn = sqlite3.connect('./pubmed.db')
	c = conn.cursor()
	with open('search_count.csv', 'w') as file:
		#writer = csv.writer(file)
		# label first row
		#writer.writerow(['']+first_keyword)
		print(['']+first_keyword)
		for k2 in second_keyword:
			c.execute("SELECT COUNT(TITLE) FROM PubMed WHERE K2=? AND K1=(SELECT K1 FROM PubMed)", (k2))
			print([k2]+c.fetchall())
			#writer.writerow([k2]+)
	conn.close()
	'''
	c.execute("SELECT * FROM PubMed")
	print(c.fetchall())
	c.execute("SELECT * FROM PubMed_Abstract")
	print(c.fetchall())
	'''

def test():
	first_keyword = ['abdominal injuries']
	second_keyword = ['C-reactive protein', 'bilirubin']
	keyword_combo = [
		('abdominal injuries', 'C-reactive protein'),
		('abdominal injuries', 'bilirubin'),
		#("hollow viscus injury", "pathogenic"), # single return and book
		#('physical trauma', 'bilirubin')
	]
	for k in keyword_combo:
		worker(*(k))
	write_result_count(first_keyword, second_keyword)

def main():
	setup_database()
	test()
	'''
	first_keyword, second_keyword, exclude = keyword()
	search(first_keyword, second_keyword, exclude)
	'''
	write_result_count(first_keyword, second_keyword)

if __name__ == '__main__':
	Entrez.email = 'david.shaw@nyu.edu'
	main()