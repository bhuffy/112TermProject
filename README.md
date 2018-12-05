# Labely - An email labeling and analysis tool for company complaints.
## 15-112 Term Project, Carnegie Mellon University

Labely is an email labeling and analysis tool for company complaints that helps them better understand common themes in emails, including key words (labels), the most important content, sentiment, and interconnections.

* Author: Bennett Huffman
* Mentor: Kusha Maharshi
* Instructors: Kelly Rivers

## Dependencies
* Python 3 (https://www.python.org/downloads/)
* NLTK (http://www.nltk.org/install.html)
    - Run `nltk.download()` upon first use of `import nltk` in file
* MatplotLib (pip install matplotlib)
* Enchant (pip install pyenchant)
* NetworkX (pip install networkx)
* Pandas (pip install pandas)

## Running Labely
Downlaod the project as a ZIP file. Select an exported CSV file of the emails you want to analyze from Gmail (may require converting from MBOX to CSV) and place it in the 'data' folder. Sample CSV files of emails can be downloaded [here](https://drive.google.com/drive/folders/1UtdwyLho-S8gaU2K0bJ6m8TXxAeJbOSV?usp=sharing). Then run the "main.py" file and follow instructions.

## Video
[Check out this example!](https://youtube.com/)

## Features
* Email labeling and graphical visualization
* Word frequency distribution and graphical visualization
* Summarization of most important content in emails (only viewable in CSV export)
* Sentiment analysis and graphical visualization over time
* CSV Export (includes all selected features in analysis)

## Navigation
* To receive instructions, click on the help screen in the top right corner.