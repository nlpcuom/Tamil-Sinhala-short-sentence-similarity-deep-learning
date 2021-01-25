### Project Installation Guide
This guide provides a step-by-step approach to setting up project.
Before you begin, be sure you have met the installation prerequisites, and then follow the instructions to run this project in your environment. 

#### Prerequisite

In order to run this project you will need to make sure that the following dependencies are installed on your system:
  - Python 3.8.*
  - tensorflow==2.4.0
  - Keras==2.3.1
  - gensim==3.8.3
  - nltk==3.5
  - numpy==1.19.5
  - pandas==1.2.0

Please find the other required dependencies on **requirements.txt** file
`pip install -r requirements.txt`

#### Directory structure

Here's a folder structure for the project:
```
Tamil-Sinhala-Short-sentence-similarity/     # Root directory.
|- data/        # For dataset files.
	|- tamil
		|- train.csv
		|- test.csv    
	|- sinhala
		|- train.csv
		|- test.csv    
|- pretrained_models/	# For pretrained word-embedding models
|- embeddings/          
|- utils/
|- preprocessing/
|- images/       		# For results output.
|- redict_si_sentence.py
|- predict_ta_sentence.py
|- train_si_malstm.py
|- train_ta_malstm.py
|- word2vec.py
|- README.md
```

#### Tamil & Sinhala stop-words setup
First, install and setup `NLTK` & `NLTK data` library and download the initial data.  
```
pip install nltk
python -m nltk.downloader stopwords
```
Download the stop-words list files available for Tamil & Sinhala  and place them into nltk data stop-words location:
- [Tamil](https://raw.githubusercontent.com/snilaxan/Tamil-Sinhala-short-sentence-similarity-deep-learning/main/stopwords/Tamil-Stopword-list?token=AAZEFDODJXODFAPAQLJRQVTABV67Y)
- [sinhala](https://raw.githubusercontent.com/snilaxan/Tamil-Sinhala-short-sentence-similarity-deep-learning/main/stopwords/Sinhala-Stopword-list?token=AAZEFDN7FOZIA53XHECLX4LABV67Q) 

Windows: `C:\Users\<USER>\AppData\Roaming\nltk_data\corpora\stopwords`

#### Pretrained word-embedding models setup
Download & place pretrained word-embedding models into `pretrained_models` directory.
 - [Tamil](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ta.300.vec.gz)
 - [Sinhala](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.si.300.vec.gz)
 
 #### Datasets for training & testing
 You should put all the dataset files to `./data` directory.
 ```
  |- ./data/tamil - for Tamil dataset
  |- ./data/sinhala - for Sinhala dataset
```

#### How to run
**Tamil**

 - For Training:- `python train_ta_malstm.py`
- For Prediction: `python predict_ta_sentence.py`

**Sinhala**
- For Training: `python train_si_malstm.py`
- For Prediction: `python predict_si_sentence.py`

#### References
