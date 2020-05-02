import requests
import urllib.request
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from Bio import Entrez
import numpy as np
import sqlite3
import traceback
from time import sleep
import os
from threading import Lock, Thread
import csv
import glob

max_retry = 5
max_batch_size = 10000

# create database
def setup_database(path):
	conn = None
	try:
		conn = sqlite3.connect(path)
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
					URL varchar(512)
				)
			''')
			c.execute('''
        		CREATE TABLE IF NOT EXISTS PubMed_Abstract(
        			TITLE varchar(512) NOT NULL REFERENCES PubMed(TITLE) ON DELETE CASCADE,
        			K1 varchar(128) NOT NULL REFERENCES PubMed(K1) ON DELETE CASCADE,
        			K2 varchar(128) NOT NULL REFERENCES PubMed(K2) ON DELETE CASCADE,
        			ABSTRACT text
        		)
			''')
			conn.commit()
			conn.close()

# extract and index keywords
def keyword(filepath='./Keywords.docx'):
    import docx
    f = docx.Document(filepath)
    first_keyword, second_keyword, exclude = {}, {}, []
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
    		container[count[0]] = t.text.strip().lower()
    		count[0] += 1
    
    return first_keyword, second_keyword, exclude

# after multithreading finished, extract results
def unpack(first_keyword, second_keyword, mat):
	# conn.isolation_level = None
	
	for i in range(len(mat)):
		for j in range(1, len(mat[0])):
			if type(mat[i][j]) != int:
				count = mat[i][j].result()
				mat[i][j] = count

	return mat

# worker function for multithreading
def worker(first, second, path):
	count = 0
	titles = []
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
				articles, abstracts = [], []
				# fetch the details for all the retrieved articles
				if len(id_list) == 1:
					count = 1
					handle = Entrez.efetch(db='pubmed',
				                           retmode='xml',
				                           id=id_list[0])

					r = Entrez.read(handle)['PubmedArticle'][0]

					# print(r)
					# check if we found a book or a research article
					if not r:
						return 0, articles, abstracts

					# collect information
					title = str(r['MedlineCitation']['Article']['ArticleTitle'])
					abstract = ' '.join([str(x) for x in r['MedlineCitation']['Article']['Abstract']['AbstractText']]) if 'Abstract' in r['MedlineCitation']['Article'] else None
					doi = None
					for x in r['PubmedData']['ArticleIdList']:
						if x.attributes['IdType'] == 'doi':
							doi = x
					url = 'http://sci-hub.tw/{}'.format(doi) if doi else None
					authors = ['{} {}'.format(x['LastName'] if 'LastName' in x else None, x['ForeName'] if 'ForeName' in x else None).strip() for x in r['MedlineCitation']['Article']['AuthorList']] if 'AuthorList' in r['MedlineCitation']['Article'] else None # [last name] [first name]
					article_type = str(r['MedlineCitation']['Article']['PublicationTypeList'][0])
					publication_year = r['PubmedData']['History'][0]['Year']
					articles.append((title, first, second, doi, ','.join(authors) if authors else authors, article_type, publication_year, url))
					abstracts.append((title, first, second, abstract))
				elif len(id_list) > 1:
					remaining = 0
					while remaining < count:
						handle = Entrez.efetch(db='pubmed',
					                           retmode='xml',
					                           id=','.join(id_list[remaining:]))
						results = Entrez.read(handle)
						
						for r in results['PubmedArticle']:
							# collect information
							title = str(r['MedlineCitation']['Article']['ArticleTitle'])
							abstract = ' '.join([str(x) for x in r['MedlineCitation']['Article']['Abstract']['AbstractText']]) if 'Abstract' in r['MedlineCitation']['Article'] else None
							doi = None
							for x in r['PubmedData']['ArticleIdList']:
								if x.attributes['IdType'] == 'doi':
									doi = str(x)
									break
							url = 'http://sci-hub.tw/{}'.format(doi) if doi else None # the downloadable links are hidden behind the onclick value with the xpath /html/body/div[1]/div[1]/ul/li
							authors = ['{} {}'.format(x['LastName'] if 'LastName' in x else None, x['ForeName'] if 'ForeName' in x else None).strip() for x in r['MedlineCitation']['Article']['AuthorList']] if 'AuthorList' in r['MedlineCitation']['Article'] else None # [last name] [first name]
							article_type = str(r['MedlineCitation']['Article']['PublicationTypeList'][0])
							publication_year = r['PubmedData']['History'][0]['Year']
							articles.append((title, first, second, doi, ','.join(authors) if authors else authors, article_type, publication_year, url))
							abstracts.append((title, first, second, abstract))
						remaining += len(results['PubmedArticle'])
				else:
					count = 0
				if len(articles) > 0:
					# write_cache(articles, abstracts)
					database = '{}'.format(path)
					setup_database(database)
					conn = sqlite3.connect(database)
					for _ in range(max_retry):
						try:
							c = conn.cursor()
							c.execute("begin")
							c.executemany('INSERT OR IGNORE INTO PubMed(TITLE, K1, K2, DOI, AUTHORS, TYPE, PUBLICATIONYEAR, URL) VALUES (?, ?, ?, ?, ?, ?, ?, ?)', articles)
							c.executemany('INSERT OR IGNORE INTO PubMed_Abstract(TITLE, K1, K2, ABSTRACT) VALUES (?, ?, ?, ?)', abstracts)
							c.execute('commit')
							articles, abstracts = [], []
							break
						except Exception as e:
							conn.rollback()
							if _ == max_retry-1:
								print('ERROR: unable to insert results of {}, {} into {}'.format(first, second, database))
							#traceback.print_exc()
					conn.close()
				print('SUCCESS "{}", "{}": {}'.format(first, second, count))
				return count
			except Exception as e:
				print('RETRY "{}", "{}": {}'.format(first, second, e))
				#traceback.print_exc()
				continue
	except Exception as e:
		print('ERROR "{}", "{}": {}'.format(first, second, e))
		pass

	return count

# search PubMed for paper and the count and search results to 
def search(first_keyword, second_keyword, exclude):
	existing_raw_databases = set(glob.glob('{}/*.db'.format(folder)))
	mat = np.full((len(first_keyword.keys()), len(second_keyword.keys())+1), None)
	num_threads = 8

	with open('{}/encoding.csv'.format(folder), 'w', encoding='utf-8') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow(['First Keyword', 'Second Keyword', 'Encoding'])
		with ThreadPoolExecutor(max_workers=num_threads) as executor:
			for i in range(mat.shape[0]):
				mat[i][0] = first_keyword[i]
				for j in range(1, mat.shape[1]):
					if '{} {}'.format(first_keyword[i], second_keyword[j-1]) in exclude:
						mat[i][j] = 0
					else:
						encoding = 'A{}_B{}'.format(i, j-1)
						path = '{}\\{}.db'.format(folder, encoding)
						writer.writerow([first_keyword[i], second_keyword[j-1], encoding])

						if path in existing_raw_databases:
							existing_raw_databases.remove(path)
							conn = sqlite3.connect(path)
							c = conn.cursor()
							mat[i][j] = c.execute('select count(title) from PubMed').fetchall()[0][0]
							conn.close()
						else:
							mat[i][j] = (executor.submit(worker, first_keyword[i], second_keyword[j-1], path))

	return mat

# write search counts to csv
def write_result_count(first_keyword, second_keyword, mat):
	np.savetxt('search_count.csv', mat, delimiter=',', header=','.join(['']+[second_keyword[i] for i in range(len(second_keyword.keys()))]), fmt=['%s'] + ['%d']*len(second_keyword.keys()), encoding='utf-8')

	# reorder
	import pandas as pd 
	df = pd.read_csv('raw_search_count.csv', header=0, index_col=0)
	sample = pd.read_excel(sample_format, header=1, index_col=1, sheet_name='Clustered data')
	sample = sample.drop(columns=['Unnamed: 0'])
	df = df.transpose()
	reordered = df.loc[sample.index, sample.columns]
	reordered.to_csv('clustered_search_count.csv')

def main():	
	first_keyword, second_keyword, exclude = keyword()
	mat = search(first_keyword, second_keyword, exclude)
	mat = unpack(first_keyword, second_keyword, mat)
	write_result_count(first_keyword, second_keyword, mat)

if __name__ == '__main__':
	folder = 'PubMed'
	if not os.path.exists(folder):
		os.mkdir(folder)

	Entrez.email = 'david.shaw@nyu.edu'
	main()