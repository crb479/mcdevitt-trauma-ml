v0

- Added functionality for scraping number of search counts for each article.
- Saved results to .csv
- Reorganized and clustered results.

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
