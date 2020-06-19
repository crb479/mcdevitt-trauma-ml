import os
import pandas as pd 

def movePDF(FILEPATH, PAPER_TYPE, STUDY_TYPE, rejected):
	dest = FILEPATH
	# When STUDY_TYPE is None, it is read as float for some reason.
	if FILEPATH and PAPER_TYPE and STUDY_TYPE:
		if type(FILEPATH) == float:
			return dest 

		source_folder = os.path.dirname(FILEPATH)

		if type(STUDY_TYPE) != float:
			dest_folder = os.path.join(dir_map[STUDY_TYPE], PAPER_TYPE)
		else:
			dest_folder = os.path.join(dir_map['unlabelled'], PAPER_TYPE)
		
		if not os.path.exists(dest_folder):
			os.mkdir(dest_folder)

		if os.path.exists(FILEPATH):
			cmd = 'move "{}" "{}"'.format(source_folder, dest_folder)
			print(cmd)
			os.system(cmd) #Windows
		
		dest = os.path.join(dest_folder, os.path.basename(FILEPATH))
		
	return dest

def main():
	summary = pd.read_csv('downloads_summary.csv')
	summary['SORTED_FILEPATH'] = summary[['FILEPATH', 'PAPER_TYPE', 'STUDY_TYPE', 'rejected']].apply(lambda x: movePDF(*x), axis=1)
	summary.to_csv('downloads_summary.csv')

if __name__ == '__main__':
	dir_map = {
		'human': 'Human Studies',
		'animal': 'Animal Studies',
		'both': 'Both',
		'unlabelled': 'Others'
	}

	for folder in dir_map.values():
		if not os.path.exists(folder):
			os.mkdir(folder)

	main()
