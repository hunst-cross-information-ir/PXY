import re
import os

def parser(text):
    if '"' in text:
        query_string=re.sub(r'"','',text)
    elif '[' in text:
        query_string=re.sub(r'[\[\]]',' ',text)
        if 'syn:' in query_string:
            query_string=re.sub(r'syn:','',query_string)
        if 'evf:' in query_string:
            query_string=re.sub(r'evf:','',query_string)
        if 'hyp' in query_string:
            query_string=re.sub(r'hyp:','',query_string)

    elif '<' in text:
        query_string=re.sub('[<>]','',text)
    elif '+' in text:
        query_string=re.sub('[+]','',text)
    elif 'EXAMPLE' in text:
        b=[]
        c=0
        for a in text:
            if a=='(':
                c=1
                continue
            elif a==')':
                continue
            if c==1:
                b.append(a)
        query_string=''.join(b)
    else:
        query_string=text
    return query_string


query_dict={} 
wf=open('G:/Openclir-Data/en3.txt','w',encoding='utf-8')  
with open('G:/Openclir-Data/OPENCLIR_2019-1A_QUERY/QUERY-EVAL/query_list.tsv','r',encoding='utf-8')as rf:
	lines=rf.readlines()
	i=0
	for line in lines:
		if i==0:
			i=i+1
			continue
		else:
			#print(line)
			text1=line.strip('\n').split('	')
			query_id=text1[0]#表示取第一个元素
			query_string=parser(text1[1])
			query_string=parser(query_string)
			query_string=parser(query_string)
			query_dict[query_id]=re.sub(',',' ',query_string)
			query_dict[query_id]=re.sub('  ',' ',query_dict[query_id])#替换两个空格
			print(query_id)
			wf.write(query_id+'	'+query_dict[query_id]+" \n")
			print(query_dict[query_id]) 


