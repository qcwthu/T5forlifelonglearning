import torch
import os
import numpy as np
import random
import csv
import pickle
def seed_everything(args):
    random.seed(args.seed)
    os.environ['PYTHONASSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def getfewshot(inpath,outpath,thislabel,fewshotnum):
    ###read from inpath
    print(thislabel)
    intrain = inpath + "/train.csv"
    intest = inpath + "/test.csv"
    alllabel = []
    trainresult = {}
    f = open(intrain, 'r')
    reader = csv.reader(f)
    for item in reader:
        #print(int(item[0]) - 1)
        if thislabel[int(item[0]) - 1] not in alllabel:
            alllabel.append(thislabel[int(item[0]) - 1])
        ###concat remains
        content = ""
        for i in range(1,len(item)):
            if i == 1:
                content = content + item[i].replace("\t"," ")
            else:
                content = content + " " + item[i].replace("\t"," ")
        if thislabel[int(item[0]) - 1] not in trainresult:
            trainresult[thislabel[int(item[0]) - 1]] = [content]
        else:
            trainresult[thislabel[int(item[0]) - 1]].append(content)
    f.close()
    # for aa in trainresult:
    #     print(aa)
    #     print(len(trainresult[aa]))
    #print(trainresult[1][0])
    testresult = {}
    f = open(intest, 'r')
    reader = csv.reader(f)
    for item in reader:
        content = ""
        for i in range(1,len(item)):
            # if "\t" in item[i]:
            #     print(item[i])
            if i == 1:
                content = content + item[i].replace("\t"," ")
            else:
                content = content + " " + item[i].replace("\t"," ")
        if thislabel[int(item[0]) - 1] not in testresult:
            testresult[thislabel[int(item[0]) - 1]] = [content]
        else:
            testresult[thislabel[int(item[0]) - 1]].append(content)
    f.close()
    # for aa in testresult:
    #     print(aa)
    #     print(len(testresult[aa]))
    # print(testresult[1][0])
    ######select few shot for train valid and test
    ###outpath
    fewtrainname = outpath + "/train.txt"
    fewvalidname = outpath + "/valid.txt"
    fewtestname = outpath + "/test.txt"

    tousetres = {}
    for key in trainresult.keys():
        if 2 * fewshotnum < len(trainresult[key]):
            thisres = random.sample(trainresult[key], 2 * fewshotnum)
        else:
            thisres = trainresult[key]
        tousetres[key] = thisres
    # for key in tousetres.keys():
    #     print(key)
    #     print(len(tousetres[key]))

    sampletestres = {}
    for key in testresult.keys():
        #sampletestnum = len(testresult[key])
        sampletestnum = 1000
        if sampletestnum < len(testresult[key]):
            thisres = random.sample(testresult[key], sampletestnum)
        else:
            thisres = testresult[key]
        sampletestres[key] = thisres

    tousetrainres = {}
    tousevalidres = {}
    for key in tousetres.keys():
        allres = tousetres[key]
        fortrain = allres[0:fewshotnum]
        forvalid = allres[fewshotnum:2 * fewshotnum]
        tousetrainres[key] = fortrain
        tousevalidres[key] = forvalid
    f = open(fewtrainname,'w')
    for key in tousetrainres.keys():
        for one in tousetrainres[key]:
            f.write(one+"\t"+key + "\n")
    f.close()

    f = open(fewvalidname, 'w')
    for key in tousevalidres.keys():
        for one in tousevalidres[key]:
            f.write(one + "\t" + key + "\n")
    f.close()

    ####test
    f = open(fewtestname, 'w')
    for key in sampletestres.keys():
        for one in sampletestres[key]:
            f.write(one + "\t" + key + "\n")
    f.close()

def getpromptembedding(model,tokenizer,promptnumber,labellist):
    t5_embedding = model.model.get_input_embeddings()
    promptinitembedding = torch.FloatTensor(promptnumber, t5_embedding.weight.size(1))
    #print(promptinitembedding)
    startindex = 0
    #print(promptinitembedding.shape)
    #print(t5_embedding.weight.shape)
    # print(tokenizer.get_vocab())
    alllabel = ["sentence classification"]
    for key in labellist.keys():
        alllabel.append(key)
        alllabel.extend(labellist[key])
    print(alllabel)
    for one in alllabel:
        encoderes = tokenizer.batch_encode_plus([one], padding=False, truncation=False, return_tensors="pt")
        touse = encoderes["input_ids"].squeeze()[:-1]
        # print(touse)
        # print(touse.shape)
        embeddingres = t5_embedding(touse).clone().detach()
        if embeddingres.shape[0] > 1:
            embeddingres = torch.mean(embeddingres, 0, keepdim=True)
        promptinitembedding[startindex] = embeddingres
        startindex += 1
        # print(embeddingres.shape)
    #print(promptinitembedding)
    # alltokens = {}
    fr = open('allnumber.pickle', 'rb')
    alltokens = pickle.load(fr)
    #print(len(alltokens))
    # print(alltokens)
    sortedalltoken = sorted(alltokens.items(), key=lambda item: item[1], reverse=True)
    # print(sortedalltoken)
    top5000 = []
    for one in sortedalltoken:
        if one[0] == 2:
            continue
        else:
            if len(top5000) < 5000:
                top5000.append(one)
            else:
                break
    #print(len(top5000))
    vocab = tokenizer.get_vocab()
    # print(vocab)
    # for one in top5000:
    #    print(one[0],"\t",one[1],"\t",tokenizer.convert_ids_to_tokens(one[0]))
    randomtokennum = promptnumber - len(alllabel)
    touse = random.sample(top5000, randomtokennum)
    print(touse)
    for one in touse:
        promptinitembedding[startindex] = t5_embedding.weight[one[0]].clone().detach()
        startindex += 1
    # print(startindex)
    # print(promptinitembedding)
    # print(t5_embedding.weight[2040])
    return promptinitembedding






