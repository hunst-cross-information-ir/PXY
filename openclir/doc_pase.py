import nltk
import os 
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
'''
def clean(_str, to_lower=True):
    """
    Cleans string from newlines and punctuation characters
    :param _str:
    :param to_lower:
    :return:
    """
    if to_lower:
        _str = _str.lower()
        # _str = _str.replace("find reports on"," ")
        # _str = _str.replace("find documents", " ")

    if _str is not None:
        _str = _str.replace("\n", " ").replace("\r", " ")
        return regex.sub(' ', _str)
    return None

def tokenize(text, language, exclude_digits=False):
    """
    Call first clean then this function.
    :param exclude_digits: whether include or exclude digits
    :param text: string to be tokenized
    :param language: language flag for retrieving stop words
    :return:
    """
    stopwords = set(stopwords.words(language))
    punctuation = set(string.punctuation)
    tokens = []
    for token in word_tokenize(text, language=language):
        if token not in stopwords and token.lower() not in stopwords and token not in punctuation and len(token) > 1:
            if exclude_digits:
                if not any(t.isdigit() for t in token):
                    tokens.append(token)
            else:
                tokens.append(token)
    return tokens

wf = open('H:/doc1.txt', 'w', encoding='utf-8')
with open('H:/OPEN_CLIR/data/OPEN-CLEF/ANALYSISN/OPENCLIR_2019-1A/ANALYSIS/text/src/MATERIAL_BASE-1A_10419561.txt','r', encoding='utf-8')as rf:
	 lines = rf.readlines()
	 for line in lines:
		 line=line.lower()
		 line = line.strip()
		 if len(line)!=0:
			 print(line)
			 stop_words = set(stopwords.words('english'))
			 word_tokens = word_tokenize(line)
			 filtered_sentence = [w for w in word_tokens if not w in stop_words]
			 filtered_sentence = []
			 english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','-']
			 for w in word_tokens:
				 if w not in stop_words: 
					 if w not in english_punctuations:
						 filtered_sentence.append(w)
						 wf.write(w+" ")
			 print(word_tokens)
			 print(filtered_sentence) 
'''			 
		
indir1='G:/Openclir-Data/parse_docs/'		 
indir='G:/Openclir-Data/OPENCLIR_2019-1A/EVAL/text/src/'
#获取文件名
names=os.listdir(indir)

for name in names:
	#print(name)
	f_name=name.strip('.txt')
	#print(f_name)
	file_dir1=indir1+name
	file_dir=indir+name
	with open(file_dir1,'w',encoding='utf-8')as wf:
		with open(file_dir,'r',encoding='utf-8')as rf:
			lines = rf.readlines()
			for line in lines:
				line=line.lower()
				line = line.strip()#去掉每一行的开头与结尾的符号
				if len(line)!=0:
					#print(line)
					stop_words = set(stopwords.words('english'))
					word_tokens = word_tokenize(line)
					#filtered_sentence = [w for w in word_tokens if not w in stop_words]
					filtered_sentence = []
					english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','-']
					for w in word_tokens:
						if w not in stop_words: 
							if w not in english_punctuations:
								filtered_sentence.append(w)
								wf.write(w+" ")
			 #print(word_tokens)
			 #print(filtered_sentence)


	
