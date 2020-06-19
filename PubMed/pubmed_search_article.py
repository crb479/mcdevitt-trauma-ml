import subprocess
from Bio import Entrez
import numpy as np
import sqlite3
import traceback
import os
import pandas as pd
import traceback
import docx
import csv
import math
from concurrent.futures import ThreadPoolExecutor
from itertools import product

def keyword(filepath):
    file = pd.read_excel(filepath)

    first_keyword = file['FIRST KEYWORD'].dropna().str.lower().str.strip()
    second_keyword = file['SECOND KEYWORD'].dropna().str.lower().str.strip()
    exclude = file['EXCLUDE'].dropna().str.lower().str.strip()
    
    return first_keyword, second_keyword, exclude

def worker(first, second):
	count, articles, abstracts = 0, [], []

	query=pmid_query.format(first, second)

	for _ in range(max_retry):
		try:
			results = Entrez.read(Entrez.esearch(db='pubmed', retmode='xml', term=query))
			break
		except Exception as e:
			if _ == max_retry-1:
				print('ERROR: unable to acquire search counts of {}, {}\n'.format(first, second), end='')
			return count, articles, abstracts

	count = int(results['Count'])
	pmids = results['IdList']

	if pmids and len(pmids) > 0:
		attempt = 0

		for i in range(int(math.ceil(count/max_batch_size))):
			try:
				handle = Entrez.efetch(db='pubmed',
			                           retmode='xml',
			                           retstart=max_batch_size*i,
			                           retmax=max_batch_size,
			                           id=','.join(pmids))
				results = Entrez.read(handle)
				
				for pmid, r in zip(pmids[max_batch_size*i:], results['PubmedArticle']):
					# collect information
					title = str(r['MedlineCitation']['Article']['ArticleTitle'])
					abstract = ' '.join([str(x) for x in r['MedlineCitation']['Article']['Abstract']['AbstractText']]) if 'Abstract' in r['MedlineCitation']['Article'] else None
					doi = None
					for x in r['PubmedData']['ArticleIdList']:
						if x.attributes['IdType'] == 'doi':
							doi = str(x)
					authors = ['{} {}'.format(x['LastName'] 
												if 'LastName' in x 
												else None, 
											  x['ForeName'] 
											  	if 'ForeName' in x 
											  	else None).strip() 
								for x in r['MedlineCitation']['Article']['AuthorList']] if 'AuthorList' in r['MedlineCitation']['Article'] else None # [last name] [first name]
					url = None # abandoned
					article_type = str(r['MedlineCitation']['Article']['PublicationTypeList'][0])
					publication_year = r['PubmedData']['History'][0]['Year']
					articles.append((pmid, title, first, second, doi, ','.join(authors) if authors else authors, article_type, publication_year, url))
					abstracts.append((pmid, title, abstract))
			except Exception as e:
				attempt += 1
				if attempt == max_retry:
					print('"{}","{}":{} --> all attempts failed\n'.format(first, second, e), end='')
					break
				else:
					print('"{}","{}":{} --> {}th attempt\n'.format(first, second, e, attempt), end='')

	print('COMPLETE: "{}", "{}" -> {}\n'.format(first, second, count), end='')

	return count, articles, abstracts

def search(first_keyword, second_keyword):
	with open(os.path.join(dest, 'articles.csv'), 'w', encoding='utf-8', newline='') as f1:
		articles_info = csv.writer(f1, delimiter=',')
		articles_info.writerow(['PMID', 'TITLE', 'K1', 'K2', 'DOI', 'AUTHORS', 'TYPE', 'PUBLICATIONYEAR', 'URL'])

		with open(os.path.join(dest, 'abstracts.csv'), 'w', encoding='utf-8', newline='') as f2:
			abstracts_info = csv.writer(f2, delimiter=',')
			abstracts_info.writerow(['PMID', 'TITLE', 'ABSTRACT'])

			with open(os.path.join(dest, 'counts.csv'), 'w', encoding='utf-8', newline='') as f3:
				count_info = csv.writer(f3, delimiter=',')
				count_info.writerow(['KEYWORD A', 'KEYWORD B', 'COUNT'])

				for a, b in product(first_keyword, second_keyword):
					count, articles, abstracts = worker(a, b)

					if len(articles) > 0:
						articles_info.writerows(articles)
						abstracts_info.writerows(abstracts)
					count_info.writerow([a, b, count])

def custom_reformat_csv():
	df = pd.read_csv(os.path.join('pubmed', 'counts.csv'))
	reordered = pd.DataFrame({'KEYWORD A': df['KEYWORD A'].unique()})
	counts = df.groupby(['KEYWORD B'], sort=False)['COUNT'].apply(list).reset_index()
	common_length = float('inf')
	for b, c in zip(counts['KEYWORD B'].unique(), counts['COUNT']):
		common_length = int(min(common_length, len(c)))
		reordered[b] = c[:common_length]
	reordered.set_index('KEYWORD A')
	reordered.to_csv(os.path.join('pubmed', 'counts_reordered.csv'))

# add argparse in the future if needed
def main(keyword_file, main_folder, email=None, num_retry=3, batch_size=100000):
	if not os.path.exists(keyword_file):
		print('Keyword file does not exist.')
		return -1

	global dest
	dest = main_folder
	global pdf_path
	pdf_path = os.path.join(main_folder, 'PDF')
	global max_retry
	max_retry = num_retry
	global max_batch_size # up to a maximum of 100,000 records
	max_batch_size = batch_size
	global pmid_query
	pmid_query = '(("{}"[Text Word]) AND ("{}"[Text Word])) AND (free full text [SB])'

	if not os.path.exists(main_folder):
		os.mkdir(dest)
	if not os.path.exists(pdf_path):
		os.mkdir(pdf_path)

	if email:
		Entrez.email = email

	first_keyword, second_keyword, exclude = keyword(keyword_file)
	for x in exclude:
		pmid_query = '(({}) NOT ("{}"[Text Word]))'.format(pmid_query, x)
	
	search(first_keyword, second_keyword)
	#reformat_csv()

	return 1

if __name__ == '__main__':
	# main('./Keywords.xlsx', 'pubmed', email='david.shaw@nyu.edu')
	custom_reformat_csv()