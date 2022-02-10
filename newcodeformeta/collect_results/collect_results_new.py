import pandas as pd
import numpy as np
import argparse
import os

# python collect_results.py --logs_dir ../models/bart-base-meta-dev-apr4 --output_file ./results_summary/bart-base-meta-dev-apr4.csv
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--logs_dir", default=["../models/T5-large-maml"], nargs="+", required=False)
    parser.add_argument("--output_file", default="./maml_hyper_tune_allres", type=str, required=False)

    args = parser.parse_args()

    onlyfinalres = "tocompareres"

    allres = {}

    for onedir in args.logs_dir:

        #print(onedir)
        allres[onedir] = {}
        df = pd.DataFrame(columns=["task", "entry", "dev_performance", "test_performance"])

        directories = os.listdir(onedir)
        for directory in sorted(directories):
            if directory.startswith("singletask"):
                task = directory[11:]
            else:
                task = directory

            if not os.path.exists(os.path.join(onedir, directory, "result.csv")):
                #print("Something wrong with task {}\n\n".format(task))
                continue

            df0 = pd.read_csv(os.path.join(onedir, directory, "result.csv"))

            devs, tests = [], []

            for idx, row in df0.iterrows():
                if row["prefix"].endswith("_best"):
                    df.loc[len(df.index)] = [task, row["prefix"][:-5], row["dev_performance"], row["test_performance"]]
                    # print(row["prefix"], row["dev_performance"], row["test_performance"])
                    devs.append(row["dev_performance"])
                    tests.append(row["test_performance"])

            if len(devs) > 0:
                df.loc[len(df.index)] = [task, "mean", np.mean(devs), np.mean(tests)]
                df.loc[len(df.index)] = [task, "std", np.std(devs), np.std(tests)]
                df.loc[len(df.index)] = ["", "", "", ""]
                allres[onedir][task] = np.mean(tests)
        df.to_csv(args.output_file)
    #print(allres)
    #print(len(allres))
    allnewres = {}
    index = 0
    firstkey = ""
    for onekey in allres.keys():
        #print(onekey)
        if index == 0:
            firstkey = onekey
        oneres = allres[onekey]
        #print(len(oneres))
        if index == 0:
            allnewres[onekey] = []
            for subkey in oneres.keys():
                allnewres[onekey].append(0.0)
        else:
            allnewres[onekey] = []
            for subkey in oneres.keys():
                allnewres[onekey].append((oneres[subkey]-allres[firstkey][subkey])/allres[firstkey][subkey])
        index += 1
    #print(allnewres)
    finalres = {}
    for onekey in allnewres.keys():
        #print(onekey)
        finalres[onekey] = np.average(allnewres[onekey])
    print(finalres)
    maxnum=-100000000.0
    maxname=""
    for onekey in finalres.keys():
        if finalres[onekey] > maxnum:
            maxnum = finalres[onekey]
            maxname = onekey
    print(maxnum)
    print(maxname)


if __name__ == "__main__":
    main()