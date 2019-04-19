

import os
'''
filelist=os.listdir('H:/submission1/')
a=[]
# print(filelist)
for file in filelist:
    with open('H:/submission1/'+file,'r',encoding='utf-8') as rf:
        text=rf.read()
        text=text.strip()
        if text=='':
            a.append(file)
for p in a:
    with open('H:/submission1/'+p,'w') as wf:
        wf.write('MATERIAL_BASE-1A_14935023	N	0.46154\n')
'''
#文件路径
indir1='G:/Openclir-Data/OPENCLIR_2019-1A/EVAL/text/src/'
#获取文件名
names1=os.listdir(indir1)
docs=[]
for name1 in names1:
	d_name=name1.strip('.txt')
	docs.append(d_name)

indir2='G:/Openclir-Data/OPENCLIR_2019-1A/EVAL/audio/src/'
#获取文件名
names2=os.listdir(indir2)
for name2 in names2:
	d_name=name2.strip('.wav')
	docs.append(d_name)
print(len(docs))

indir='G:\Openclir-Data\submission1/'
#获取文件名
names=os.listdir(indir)
for name in names:
	#print(name)
	f_name=name.strip('.tsv')
	#print(' 1')
	#print(f_name)
	file_dir=indir+name
	#wf = open(file_dir,'a+',encoding='utf-8')
	with open(file_dir,'a+',encoding='utf-8')as f:
		lines=f.readlines()
	
		i = 0
		for line in lines:
			text=line.strip('\n').split()
			doc_id=text[0]
		for doc in docs:
			
			f.write(doc+'	'+'N'+'	'+'0.00000'+'\n')
		print(1)
			
		


