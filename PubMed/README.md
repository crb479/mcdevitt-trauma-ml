**INPUT**

*keywords.xlsx*

    Lists of keywords and phrases to exclude.

*oa_file_list.csv*

    Index file from PubMed. It specifies the directories of PDFs on their FTP server, and the corresponding license types (commercial vs noncommercial).

*exclusion_criteria.xlsx*

    Listed of accepted keywords for human-related and animal-related studies, accepted paper types, and study designs.

**SCRIPT**

*pubmed_downloaded_openaccess.py*

    Combine columns of articles.csv and oa_file_list.csv for a general overview of license information for articles that were found by the search functions and are available on PubMed FTP. The final results are stored in OpenAccess.csv.

*pubmed_ftp_download.py*

    Download and unpack archives from PubMed FTP that are listed on OpenAccess.csv.

*pubmed_ftp_download_summary.py*

    Find and examine each downloaded PDFs in ./PDF to compute various statistics used to exclude/include PDFs. The criteria of exclusion are specified in exclusion_criteria.xlsx.

*pubmed_movePDF.py*

    Move folders of PDFs to their respective destinations using Windows commands. The new paths are added as a new column to downloads_summary.csv.

*pubmed_search_article.py*

    Find and record article information and abstracts for each articles found, using PubMed API from Bio.Entrez. The searches are strict in that the returned articles must contain the keywords in their entirety. The keywords used in the search are specified in keywords.xlsx. The results are stored in articles.csv, abstracts.csv, and counts.csv.

    custom_reformat_csv() changes how counts.csv looks and stores the changes in counts_reordered.csv.

*pubmed_human_study_design_split.py*
    
    Label the human studies with study designs based on the design keywords from exclusion_criteria.xlsx. The results are stored in human_study_desgin_label.csv.

**OUPUT**

*PDF.zip*

    An archive of all downloaded PDFs from PubMed FTP service. This is manually added.

*PDF*

    (After pubmed_ftp_download.py) A collection of the PDFs, which are stored in folders named with their PubMed IDs. Each folder contains a copy of the correponding PDF and pictures of the tables and images in the PDFs, along with a .nxml file.

    (After pubmed_separate_downloads.py) The remaining downloaded folders that *DO NOT* contain PDFs. These folders also have missing filepaths in downloads_summary.csv

*Human Studies*

    A collection of folders of PDFs that contain human-related keywords specified in exclusion_criteria.xlsx. Folders are further divided based on their paper types.

*Animal Studies*

    A collection of folders of PDFs that contain animal-related keywords specified in exclusion_criteria.xlsx. Folders are further divided based on their paper types.

*Both*

    A collection of folders of PDFs that contain both human-related keywords and animal-related keywords specified in exclusion_criteria.xlsx. Folders are further divided based on their paper types.

*Others*

    A collection of folders of PDFs that contain neither human-related keywords nor animal-related keywords specified in exclusion_criteria.xlsx. Folders are further divided based on their paper types.

*articles.csv*

    Information of articles returned by PubMed API.

*abstracts.csv*

    Abstracts for articles returned by PubMed API.

*counts.csv*

    Counts of articles returned by PubMed API for each keyword combination.

*counts_reordered.csv*

    Counts of articles returned by PubMed API for each keyword combination.

*downloads_summary.csv*

    Aggregated information for downloaded PDF folders with relevant license information and statistics for rejection.

*human_study_desgin_label.csv*

    Study design labels for each PDF classified as human study.
