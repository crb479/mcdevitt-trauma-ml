import requests
import urllib.request
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from Bio import Entrez
import numpy as np
import os
import traceback

max_retry = 3

# extract and index keywords
def keyword(filepath='./Keywords for Systematic Review Analysis for Trauma by DV-01-29-2020.docx'):
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
def unpack(mat):
	for i in range(mat.shape[0]):
		for j in range(1, mat.shape[1]):
			if type(mat[i][j]) != int:
				mat[i][j] = mat[i][j].result()
	return mat

# worker function for multithreading
def worker(first, second, filepath):
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
				                           id=id_list[0])
					r = Entrez.read(handle)['PubmedArticle']
					if r:
						with open('{}/index.txt'.format(filepath), 'w', encoding='utf-8') as f:
							with open('{}/download_links.txt'.format(filepath), 'w', encoding='utf-8') as f2:
								remaining = 0
								while remaining < count:
									handle = Entrez.efetch(db='pubmed',
								                           retmode='xml',
								                           id=','.join(id_list[remaining:]))
									results = Entrez.read(handle)
									for r in results['PubmedArticle']:
										f.write("<TIL>{}\n<ABS>{}\n<PMID>{}\n<DOI>{}\n".format(
											r['MedlineCitation']['Article']['ArticleTitle'],
											' '.join([str(x) for x in r['MedlineCitation']['Article']['Abstract']['AbstractText']]) if 'Abstract' in r['MedlineCitation']['Article'] else '',
											r['PubmedData']['ArticleIdList'][0],
											str(r['MedlineCitation']['Article']['ELocationID'][-1]) if 'ELocationID' in r['MedlineCitation']['Article'] and r['MedlineCitation']['Article']['ELocationID'] else ''
										))
										f2.write('{}: {}\n'.format(
											r['MedlineCitation']['Article']['ArticleTitle'], 
											'sci-hub.tw/{}'.format(str(r['MedlineCitation']['Article']['ELocationID'][-1])) if 'ELocationID' in r['MedlineCitation']['Article'] and r['MedlineCitation']['Article']['ELocationID'] else ''
										))
									remaining += len(results['PubmedArticle'])
					else:
						count = 0
				else:
					count = 0
				print('SUCCESS: "{}, {}" -> {}\n'.format(first, second, count), end='')
				return count
			except Exception as e:
				print('WARNING: Retrying keywords "{}" and "{}": {}\n'.format(first, second, e), end='')
				# traceback.print_exc()
				continue
		print('ERROR: "{}" and "{}"\n'.format(first, second), end='')
	except Exception as e:
		print('ERROR: "{}" and "{}": {}\n'.format(first, second, e), end='')
		# traceback.print_exc()

	return count

# search PubMed for paper and the count and search results to 
def search(first_keyword, second_keyword, exclude):
	mat = np.full((len(first_keyword.keys()), len(second_keyword.keys())+1), None)

	with open('PubMed/_index.txt', 'w', encoding='utf-8') as f:
		with ThreadPoolExecutor(max_workers=3) as executor:
			for i in range(mat.shape[0]):
				if not os.path.exists('PubMed/A{}'.format(i)):
					os.mkdir('PubMed/A{}'.format(i))
				mat[i][0] = first_keyword[i]
				for j in range(1, mat.shape[1]):
					f.write('"{}", "{}": A{}/B{}\n'.format(first_keyword[i], second_keyword[j-1], i, j-1))
					# check if combination should be excluded
					if '{} {}'.format(first_keyword[i], second_keyword[j-1]) in exclude:
						mat[i][j] = 0
					else:
						if not os.path.exists('PubMed/A{}/B{}'.format(i, j-1)):
							os.mkdir('PubMed/A{}/B{}'.format(i, j-1))
						mat[i][j] = (executor.submit(worker, first_keyword[i], second_keyword[j-1], 'PubMed/A{}/B{}'.format(i, j-1)))

	return mat

# write search counts to csv
def write_result_count(first_keyword, second_keyword, mat):
	import pandas as pd 
	np.savetxt('raw_search_count.csv', mat, delimiter=',', header=','.join(['']+[second_keyword[i] for i in range(len(second_keyword.keys()))]), fmt=['%s'] + ['%d']*len(second_keyword.keys()), encoding='utf-8')
	df = pd.read_csv('raw_search_count.csv', header=0, index_col=0)
	sample = pd.read_excel(sample_format, header=1, index_col=1, sheet_name='Clustered data')
	sample = sample.drop(columns=['Unnamed: 0'])
	df = df.transpose()
	reordered = df.loc[sample.index, sample.columns]
	reordered.to_csv('clustered_search_count.csv')

def test():
	if not os.path.exists('test'):
		os.mkdir('test')
	keyword_combo = [
		("hollow viscus injury", "pathogenic")
	]
	for k in keyword_combo:
		if not os.path.exists('test/{}_{}'.format(k[0], k[1])):
			os.mkdir('test/{}_{}'.format(k[0], k[1]))
		worker(*(k), 'test/{}_{}'.format(k[0], k[1]))

def main():
	first_keyword, second_keyword, exclude = keyword()
	if not os.path.exists('PubMed'):
		os.mkdir('PubMed')
	mat = search(first_keyword, second_keyword, exclude)
	mat = unpack(mat)
	write_result_count(first_keyword, second_keyword, mat)

if __name__ == '__main__':
	Entrez.email = 'david.shaw@nyu.edu'
	main()