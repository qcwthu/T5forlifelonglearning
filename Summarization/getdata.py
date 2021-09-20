from tqdm import tqdm
import os
os.environ['TRANSFORMERS_CACHE'] = '/data/qin/cache/'
import datasets
def write_to_txt(l, path):
    with open(path, "w") as f:
        for i in tqdm(range(len(l))):
            f.write(l[i] + "\n")
            # if i >= 10:
            #     raise Exception
data_path = "data"

###cnndm
dataset_name = "cnndm"
sets = [("validation", "valid"), ("test", "test"), ("train", "train")]
contents = [("article", "text"), ("highlights", "summary")]
cnndm = datasets.load_dataset("cnn_dailymail", "3.0.0", cache_dir="/data/qin/cache/")

###xsum
# dataset_name = "xsum"
# sets = [("validation", "valid"), ("test", "test"), ("train", "train")]
# contents = [("document", "text"), ("summary", "summary")]
# cnndm = datasets.load_dataset("xsum", cache_dir="/data/qin/cache/")

if not os.path.exists(data_path):
    os.mkdir(data_path)
if not os.path.exists(data_path + "/" + dataset_name):
    os.mkdir(data_path + "/" + dataset_name)

for x in sets:
    (set, set_name) = x
    alltouse = {}
    for y in contents:
        (content, content_name) = y
        #print(set, set_name, content, content_name)
        dataset = cnndm[set]
        #print(dataset[0])
        text = [x[content].replace("\n"," ").replace("\t"," ").strip(' ') for x in dataset]
        alltouse[content_name] = text
        #print(set, len(text))
        path = data_path + "/" + dataset_name + "/{}_{}.txt".format(set_name, content_name)
        write_to_txt(text, path)
    for key in alltouse.keys():
        print(key,len(alltouse[key]))
    filename = data_path + "/" + dataset_name + "/{}.txt".format(set_name)
    thistext = alltouse['text']
    thissummary = alltouse['summary']
    f = open(filename,'w')
    for i in range(len(thistext)):
        f.write(thistext[i]+"\t"+thissummary[i]+"\n")
    f.close()
