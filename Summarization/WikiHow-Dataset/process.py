'''
This code is used to create article and summary files from the csv file.
The output of the file will be a directory of text files representing seoarate articles and their summaries.
Each summary line starts with tag "@summary" and the article is followed by "@article".
'''
import pandas as pd
import os
import re

# read data from the csv file (from the location it is stored)
Data = pd.read_csv(r'wikihowAll.csv')
Data = Data.astype(str)
rows, columns = Data.shape

# The path where the articles are to be saved
path = "data"
if not os.path.exists(path):
    os.makedirs(path)

# go over the all the articles in the data file
allnum = 0
allarticleres = []
allabstractres = []
for row in range(rows):
    abstract = Data.loc[row,'headline']      # headline is the column representing the summary sentences
    article = Data.loc[row,'text']           # text is the column representing the article
    #  a threshold is used to remove short articles with long summaries as well as articles with no summary
    if len(abstract) < (0.75*len(article)):
        abstract = abstract.replace(".,",".").replace("\n"," ").replace("\t"," ").strip(' ')
        article = re.sub(r'[.]+[\n]+[,]',".\n", article)
        article = article.replace("\n"," ").replace("\t"," ").strip(' ')
        allabstractres.append(abstract)
        allarticleres.append(article)
        allnum += 1
print(allnum)
assert len(allabstractres) == len(allarticleres)
tosplit = [0.8,0.9]

trainarticle = []
trainabstract = []
validarticle = []
validabstract = []
testarticle = []
testabstract = []

trainarticle = allarticleres[0:int(len(allarticleres)*tosplit[0])]
validarticle = allarticleres[int(len(allarticleres)*tosplit[0])+1:int(len(allarticleres)*tosplit[1])]
testarticle = allarticleres[int(len(allarticleres)*tosplit[1])+1:]
print(len(trainarticle))
print(len(validarticle))
print(len(testarticle))

trainabstract = allabstractres[0:int(len(allabstractres)*tosplit[0])]
validabstract = allabstractres[int(len(allabstractres)*tosplit[0])+1:int(len(allabstractres)*tosplit[1])]
testabstract = allabstractres[int(len(allabstractres)*tosplit[1])+1:]
print(len(trainabstract))
print(len(validabstract))
print(len(testabstract))


f = open("train.txt",'w')
for i in range(len(trainarticle)):
    f.write(trainarticle[i] + "\t" + trainabstract[i] + '\n')
f.close()

f = open("valid.txt",'w')
for i in range(len(validarticle)):
    f.write(validarticle[i] + "\t" + validabstract[i] + '\n')
f.close()

f = open("test.txt",'w')
for i in range(len(testarticle)):
    f.write(testarticle[i] + "\t" + testabstract[i] + '\n')
f.close()