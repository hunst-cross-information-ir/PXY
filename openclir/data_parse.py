import re
import os
from nltk.tokenize import word_tokenize, wordpunct_tokenize


def parser(text):
    if '"' in text:
        query_string = re.sub(r'"', '', text)
    elif '[' in text:
        query_string = re.sub(r'[\[\]]', ' ', text)
        if 'syn:' in query_string:
            query_string = re.sub(r'syn:', '', query_string)
        if 'evf:' in query_string:
            query_string = re.sub(r'evf:', '', query_string)
        if 'hyp' in query_string:
            query_string = re.sub(r'hyp:', '', query_string)

    elif '<' in text:
        query_string = re.sub('[<>]', '', text)
    elif '+' in text:
        query_string = re.sub('[+]', '', text)
    elif 'EXAMPLE' in text:
        b = []
        c = 0
        for a in text:
            if a == '(':
                c = 1
                continue
            elif a == ')':
                continue
            if c == 1:
                b.append(a)
        query_string = ''.join(b)
    else:
        query_string = text
    return query_string


english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '<', '>', '+', '+,', '],', '+[', '-']

query_dict = {}
# wf=open('/Users/zhoudong/Experiments/CLIR/openclir/enq.txt','w',encoding='utf-8')
with open('H:/OPEN_CLIR/data/OPEN-CLEF/QUERY-DEV/OPENCLIR_2019-1A/QUERY-DEV/query_list.tsv', 'r', encoding='utf-8')as rf:
    lines = rf.readlines()
    for line in lines[1:]:#从第二行开始
        text1 = line.strip('\n').split('	')
        query_id = text1[0]  # 表示取第一个元素
        query_string = parser(text1[1])
        # print(query_string)
        query = wordpunct_tokenize(query_string)
        query_string = ''
        for q in query:
            if q not in english_punctuations:
                query_string = query_string + ' ' + q
        query_string = query_string.strip()

        print(query_id)
        print(query_string)
        #wf.write(query_id+'    '+query_dict[query_id]+" \n")
        # print(query_dict[query_id])
