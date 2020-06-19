import os
import pandas as pd 
from shutil import unpack_archive, rmtree
from ftplib import FTP
from multiprocessing import Process

# DOWNLOAD archive from PubMed FTP server
def download(path, filename, accessionID):
	dest = '{}.tar.gz'.format(os.path.join(PDF_dir, filename))
	print(path, filename, dest)
	with open(dest, 'wb') as fp:
		ftp.retrbinary('RETR {}'.format(path), fp.write)
	Process(target=extract, args=(dest, accessionID,)).start() # a process pool is a better idea if archives have a lot of files in general.

# EXTRACT all files from the archive and store under the folder named with the PMID and DELETE the archive
def extract(archive, accessionID):
	parent = archive[:-7]
	unpack_archive(archive, parent, format='gztar')
	for file in os.listdir(os.path.join(parent, accessionID)):
		os.system('move {} {}'.format(os.path.join(parent, accessionID, file), parent))
	rmtree(os.path.join(parent, accessionID))
	os.system('del {}'.format(archive))

def main():
	index[['File', 'PMID', 'Accession ID']].apply(lambda x: download(x[0], x[1], x[2]), axis=1)
	ftp.quit()

if __name__ =='__main__':
	index = pd.read_csv('OpenAccess.csv')
	index = index[['File', 'PMID', 'Accession ID']]
	index['PMID'] = index['PMID'].astype(str)

	PDF_dir = 'PDF'
	
	if not os.path.exists(PDF_dir):
		os.mkdir(PDF_dir)

	# login info for FTP
	host = 'ftp.ncbi.nlm.nih.gov'
	username = 'anonymous'
	password = 'david.shaw@nyu.edu' # use email if username is "anonymous"

	ftp = FTP(host=host, user=username, passwd=password)
	ftp.cwd('pub/pmc/')
	print(ftp.pwd())

	main()