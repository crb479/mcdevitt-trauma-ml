import pandas as pd
import os
import fitz
import re

# i forgot to add pmid into the sorted filepaths, so here is a patch.
def fix_sorted_filepath():
	def fix_one_path(PMID, old_path):
		#print(PMID, old_path, end=' -> ')
		if pd.isnull(old_path):
			return None
		parts = old_path.split('\\')
		parts.insert(2, str(PMID))
		new_path = '\\'.join(parts)
		#print(new_path)
		return new_path

	studies = pd.read_csv(index, index_col=0)
	studies['SORTED_FILEPATH'] = studies[['PMID', 'SORTED_FILEPATH']].apply(lambda x: fix_one_path(*x), axis=1)
	studies.to_csv(index)

def label_human_study_design():
	study_design_types = pd.read_excel(exclusion_criteria)['study_designs'].dropna().str.lower().values
	studies = pd.read_csv(index, index_col=0)

	studies['STUDY_DESIGN_TYPE'] = ""
	human_studies = studies[studies['STUDY_TYPE'] == 'human'].copy()

	def get_study_design_type(path):
		print(path, end=' -> ')
		if not path:
			print('None')
			return None
		elif not os.path.exists(path):
			# some of the downloaded archives' pdf files have different titles (article titles) from the ones returned by the search
			# it is later discovered that this discrepency led to 
			print('ERROR')
			return 'ERROR'
		else:
			doc = fitz.open(path)
			design_keywords = set()

			for page in doc:
				text = page.getTextPage().extractText().lower()
				# print(text)
				for word in study_design_types:
					if word in text:
						design_keywords.add(word)
			doc.close()
			print(list(design_keywords))
			return list(design_keywords)

	human_studies['STUDY_DESIGN_TYPE'] = human_studies.SORTED_FILEPATH.apply(lambda x: get_study_design_type(x))
	human_studies.to_csv('human_study_desgin_label.csv')

def move_human_study_pdf():
	pass

def main():
	global human_studies_path
	global exclusion_criteria
	global index

	human_studies_path = './Human Studies'
	exclusion_criteria = './exclusion_criteria.xlsx'
	index = './downloads_summary.csv'

	# fix_sorted_filepath()
	# label_human_study_design()
	fix_missing_pdf()
	# move_human_study_pdf()

if __name__ == '__main__':
	main()