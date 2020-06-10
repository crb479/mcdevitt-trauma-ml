How to run:

	1. customize parameters in pubmed_article_info.py for filepaths
	2. run "python pubmed_article_info.py"
	3. (optional) use SQLiteStudio to convert files from .csv to .db
  
V3

- Changed file structures for storing results
- Changed keyword extension to .xlsx
- Changed storage format of article information to csv

V2

- Changed storage to sqlite .db
- Added scripts for merging databases and cleaning data

v1

- Added functionality for scraping DoI, PubMedID, and abstract for each article.
- Added downloadable links for articles that have DoI.
- Modified how article information is saved. 

-- Each folder in PubMed/A[#]/B[#]/ contains index.txt and download_links.txt

	index.txt contains a brief description for each article.
		<TIL> / title
		<ABS> / abstract
		<PMID> / PubMed ID
		<DOI> / Document ID
	  
-- download_links.txt contains the downloadable urls for each article title if they have doi's.
      	
	[title]: [url]

v0

- Added functionality for scraping number of search counts for each article.
- Saved results to .csv
- Reorganized and clustered results.
