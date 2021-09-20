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
    alllabel = ["summarization"]
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

def get_grad_params(model):
    grad_params = []
    for name,param in model.named_parameters():
        #ifuseewc = True
        ifuseewc = False
        if "model.decoder.block.21" in name or "model.decoder.block.22" in name or "model.decoder.block.23" in name:
        #if "model.decoder.block.22" in name or "model.decoder.block.23" in name:
           ifuseewc = True
        if ifuseewc and param.requires_grad:
            grad_params.append(param)
    return grad_params

def copy_param_data(params):
    copy_params = []
    for param in params:
        copy_params.append(param.data.clone())
    return copy_params

def param_loss(model, means, fishers, p_lambda, args):
    grad_params = get_grad_params(model)
    loss = torch.tensor(0.0).to(args.device)
    for i, param in enumerate(grad_params):
        loss += (p_lambda*fishers[i]*(param-means[i])**2).sum()
    return loss

def gen_fisher(model, args, pre_dataloader):
    fisher = None
    model.eval()
    for step, batch in enumerate(pre_dataloader):
        model.zero_grad()
        inputs = {"input_ids": batch[0].to(args.device), "attention_mask": batch[1].to(args.device),
                  "target_ids": batch[2].to(args.device), "target_mask": batch[3].to(args.device)}
        loss = model(inputs)
        finalloss = loss
        finalloss.backward()
        grad_params = get_grad_params(model)
        if fisher is None:
            fisher = [param.grad.abs_()/len(pre_dataloader)
                         for param in grad_params]
        else:
            fisher = [fisher[i]+param.grad.abs_()/len(pre_dataloader)
                         for i,param in enumerate(grad_params)]
    model.train()
    return fisher




