import pandas as pd 
from glob import glob
import os
import fitz # PyMuPDF
import re

# determine whether a PDF should be excluded
def get_rejection_status(paper_type, word_count):
	status = False
	log = []

	if not paper_type or paper_type not in accepted_paper_types:
		status = True
		log.append('paper_type')
	if not word_count or word_count < threshold:
		status = True
		log.append('trauma_count')

	if len(log) == 0:
		log = None

	return status, log

# categorize a PDF as human studies, animal studies, or both
def get_subject_type(path):
	if not path:
		return None

	doc = fitz.open(path)
	isHuman, isAnimal = False, False

	for page in doc:
		text = page.getTextPage().extractText().lower()
		if isHuman and isAnimal:
			break
		if not isHuman:
			for word in human_keywords:
				if re.search(r'(?<=[^\w]){}(?=[^\w])'.format(word), text):
					isHuman = True
		if not isAnimal:
			for word in animal_keywords:
				if re.search(r'(?<=[^\w]){}(?=[^\w])'.format(word), text):
					isAnimal = True
	doc.close()

	if isHuman and isAnimal:
		return 'both'
	if isAnimal:
		return 'animal'
	elif isHuman:
		return 'human'
	else:
		return None

# count the number of times a target word is mentioned
def get_word_count(path):
	if not path:
		return None

	doc = fitz.open(path)
	num_trauma = 0
	for page in doc:
		text = page.getTextPage().extractText().lower()
		# "(?<=[^\w]){}(?=([^\w]|es[^\w]|s[^\w]))" for plural form of word
		num_trauma += len(re.findall(r'(?<=[^\w]){}(?=[^\w])'.format(word), text)) # only singular form
	doc.close()
	return num_trauma

def find_pdf(path):
	pdfs = glob(os.path.join(path, '*.pdf'))

	# in some rare scenarios, an archive can contain more than one PDF.
	if len(pdfs) == 0:
		pdf = None
	elif len(pdfs) > 1:
		title = index.TITLE[index.PMID == pmid].values[0][:-1].lower() # PubMed API adds a period to the end of each title
		for x in pdfs:
			doc = fitz.open(x)
			first_page = doc[0].getTextPage().extractText().lower().replace('\n', ' ')
			doc.close()
			if first_page.find(title) > -1:
				pdf = x
				break
	else:
		pdf = pdfs[0]

	return pdf

def create_summary(in_path, out_path):
	index = pd.read_csv(in_path)[['PMID', 'TITLE', 'TYPE']].drop_duplicates(subset='PMID')

	pmids = []
	pdf_paths = []
	study_types = []
	trauma_counts = []
	rejection_status = []
	rejection_log = []

	for path in glob(os.path.join(PDF_ROOT, '*')):
		pmid = int(os.path.basename(path))
		paper_type = index.TYPE[index.PMID == pmid].values[0]
		pdfs = glob(os.path.join(path, '*.pdf'))

		# in some rare scenarios, an archive can contain more than one PDF.
		pdf = find_pdf(path)
		study_type = get_subject_type(pdf)
		trauma_count = get_word_count(pdf)
		status, log = get_rejection_status(paper_type, trauma_count)

		pmids.append(pmid)
		pdf_paths.append(pdf)
		study_types.append(study_type)
		trauma_counts.append(trauma_count)
		rejection_status.append(status)
		rejection_log.append(log)

		print(pmid, pdf, study_type, trauma_count, status, log)
		
	summary = pd.DataFrame({
		'PMID': pmids,
		'FILEPATH': pdf_paths,
		'STUDY_TYPE': study_types,
		'trauma_counts': trauma_counts,
		'rejected': rejection_status,
		'rejection_log': rejection_log
	})
	summary = summary.merge(index, on='PMID', how='left')
	summary = summary.rename(columns={'TYPE': 'PAPER_TYPE'})
	summary[['PMID', 'TITLE', 'FILEPATH', 'PAPER_TYPE', 'STUDY_TYPE', 'trauma_counts', 'rejected', 'rejection_log']].to_csv(out_path)
	
if __name__ == '__main__':
	PDF_ROOT = 'PDF'
	exclusion_criteria = pd.read_excel('exclusion_criteria.xlsx')

	human_keywords = exclusion_criteria.human_keywords.values
	animal_keywords = exclusion_criteria.animal_keywords.values
	accepted_paper_types = exclusion_criteria.accepted_types.values

	word='trauma'
	threshold = 3

	create_summary('articles.csv', 'downloads_summary.csv')