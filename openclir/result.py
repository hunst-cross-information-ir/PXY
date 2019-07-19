import re
import os
#创建文件
def create_tsv(name):
	path='H:\OPEN_CLIR\submission1\\'
	fullpath=path+name+'.tsv'
	f=open(fullpath,'w')

#create_tsv(query)
'''
with open('G:/Openclir-Data/query1.txt','r',encoding='utf-8') as rf:
	lines=rf.readlines()
	i=0
	for line in lines:
		i=i+1
		text=line.strip('\n').split()
		query_id=str(text[0])#表示取第一个元素
		print(query_id)
		create_tsv(query_id)
	print(i)
		
'''				
#文件路径
#indir='G:/Openclir-Data/submission/'
indir='H:/OPEN_CLIR/use_CLIR_tools/CLIR_tools/mydata/submission/'
#获取文件名
names=os.listdir(indir)
#print(names)


for name in names:
	#print(name)
	f_name=name.strip('.tsv')
	#print(f_name)
	file_dir=indir+name
	with open(file_dir,'w',encoding='utf-8')as f:
		with open('H:/open_clir/dict_BM25F1.txt','r',encoding='utf-8')as rf:
			lines=rf.readlines()
			for line in lines:
				text=line.strip('\n').split()
				query_id=text[0]
				score=float(text[2])
				if f_name==query_id:
					if score>0.95:
						f.write(text[1]+'	'+'Y'+'	'+text[2]+'\n')
					else:
						f.write(text[1]+'	'+'N'+'	'+text[2]+'\n')



