import torch
import os
import numpy as np
import random
import csv
import pickle
from collections import Counter
import string
import re
def seed_everything(args):
    random.seed(args.seed)
    os.environ['PYTHONASSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def getfewshot(inpath,outpath,fewshotnum):
    ###read from inpath
    intrain = inpath + "/train.txt"
    invalid = inpath + "/valid.txt"
    intest = inpath + "/test.txt"
    alltrainres = []
    allvalidres = []
    alltestres = []

    f = open(intrain,'r')
    while True:
        oneline = f.readline().strip()
        if not oneline:
            break
        alltrainres.append(oneline)
    f.close()

    f = open(invalid, 'r')
    while True:
        oneline = f.readline().strip()
        if not oneline:
            break
        allvalidres.append(oneline)
    f.close()

    f = open(intest, 'r')
    while True:
        oneline = f.readline().strip()
        if not oneline:
            break
        alltestres.append(oneline)
    f.close()

    ######select few shot for train valid and test
    ###outpath
    fewtrainname = outpath + "/train.txt"
    fewvalidname = outpath + "/valid.txt"
    fewtestname = outpath + "/test.txt"

    tousetrainres = random.sample(alltrainres, fewshotnum)
    tousevalidres = random.sample(allvalidres, fewshotnum)
    testnum = 500
    tousetestres = random.sample(alltestres, testnum)

    f = open(fewtrainname,'w')
    for one in tousetrainres:
        f.write(one + "\n")
    f.close()

    f = open(fewvalidname, 'w')
    for one in tousevalidres:
        f.write(one + "\n")
    f.close()

    ####test
    f = open(fewtestname, 'w')
    for one in tousetestres:
        f.write(one + "\n")
    f.close()

def getpromptembedding(model,tokenizer,promptnumber,taskname):
    t5_embedding = model.model.get_input_embeddings()
    promptinitembedding = torch.FloatTensor(promptnumber, t5_embedding.weight.size(1))
    #print(promptinitembedding)
    startindex = 0
    #print(promptinitembedding.shape)
    #print(t5_embedding.weight.shape)
    # print(tokenizer.get_vocab())
    #alllabel = ["summarization"]
    alllabel = ["question answering"]
    alllabel.append(taskname)
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

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def qa_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    # prediction_tokens = prediction.split()
    # ground_truth_tokens = ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1



