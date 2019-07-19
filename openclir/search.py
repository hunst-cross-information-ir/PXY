#-*-coding:GBK -*-
import re
from whoosh.fields import *
import os
import sys
from whoosh.index import create_in
from whoosh.query import *
import gc
from whoosh.qparser import 	QueryParser
from whoosh import qparser
from whoosh import scoring
#from WeightingModel import LM
from whoosh.scoring import WeightingModel,BaseScorer,WeightLengthScorer
from gensim.models import Word2Vec


schema=Schema(id=TEXT(stored=True),content=TEXT(stored=True))

if not os.path.exists("index_file"):
	os.mkdir("index_file")
ix=create_in("index_file",schema)
writer=ix.writer()


'''
query_dict={}
with open('H:/OPEN_CLIR/sw1.txt','r',encoding='utf-8') as rf:
    lines=rf.readlines()
    i=0
    for line in lines:
        text1=line.strip().split()
        query_id=text1[0]#表示取第一个元素
        query_string=text1[1]#表示取第二个元素
        #print(query_id)
        print(query_id+query_string)
        i=i+1
        
        query_dict[query_id]=query_string
print(i) 
'''

query_dict={}
with open('H:/open_clir/sw3.txt','r',encoding='utf-8') as rf:
    lines=rf.readlines()
    i=0
    for line in lines:
        text1=line.strip().split('	')
        query_id=text1[0]#表示取第一个元素
        query_string=text1[1].lower()#表示取第二个元素
        #print(query_id)
        print(query_id+query_string)
        i=i+1
        
        query_dict[query_id]=query_string
print(i) 

#文件路径
indir='H:/OPEN_CLIR/parse_docs/'
#获取文件名
names=os.listdir(indir)
#print(names[0])
#定义一个空字典
dic_all={}
i=0
for name in names:
	#print(name)
	i=i+1
	file_dir=indir+name
	with open(file_dir,'r',encoding='utf-8')as f:
		text=f.read()
	#print(text)
	index=name.rfind('.')
	name=name[:index]
	#print(name)
	file_id=name
	#print(file_id)
	dic_all[file_id]=text
	#print(dic_all[file_id])
	writer.add_document(id=file_id,content=text)
	
writer.commit()
print(i)


results_dict={}
with open ('H:/open_clir/dict_BM25F1.txt','w',encoding='utf-8') as wf:
	#searcher=ix.searcher()#默认为BM25
	#weighting=scoring.BM25F()
	#searcher=ix.searcher(weighting=scoring.TF_IDF())
	#searcher=ix.searcher(weighting=LM())
	with ix.searcher(weighting=scoring.BM25F(B=0.75,k1=1.5)) as searcher:
		og = qparser.OrGroup.factory(0.9)
		parser=qparser.QueryParser("content",schema=ix.schema,group=og)
		for query in query_dict.items():
			myquery=parser.parse(query[1])
			print(myquery)
			#print(query[0])
			
			results=searcher.search(myquery,limit=None)
			query1=[]
			
			i=0
			for result in results:
				i=i+1
				query1.append(result.score)
			#print(query1)
	
			if i>1:
				max=query1[0]
				min=query1[0]
				for q in query1[1:]:
					if(max<q):
						max=q
					if(min>q):
						min=q
				#print(max)
				for result in results:
					if(max!=min):
						score=(result.score-min)/(max-min)
						wf.write(query[0]+'	'+result['id']+'	'+str('%.5f' % score)+' BM25F\n')
			else:
				for result in results:
					#score=result.score/result.score
					score=result.score
					wf.write(query[0]+'	'+result['id']+'	'+str('%.5f' % score)+' BM25F\n')
			
	

