import codecs
from codecs import decode
import string

def loadLabels(filename, charset="utf-8"):
    f = codecs.open(filename, "r", charset, "ignore")
    labels = dict()
    for line in f:
        if(not "\t" in line):
            continue
        (k, v) = line.split("\t")
        labels[k] = string.atof(v)
    return labels

def filterDocuments(filename, aSet, charset="utf-8"):
    f = open(filename, "r")
    fout=file("data/test/traindata","w")
    for line0 in f:
        try:
            line = decode(line0, charset, 'ignore')
        except:
            continue
        tokens = line.split("\t")
        idStr = tokens[0]
        if idStr in aSet.keys():
            fout.write(line0)
    f.close()
    fout.close()

def mergeLabelData(metadata, label, output):
    f = open(label, "r")
    
    labels = dict()
    
    for line0 in f:
        try:
            line = decode(line0, "gbk", 'ignore').strip()
        except:
            continue
        tokens = line.split(" ")
        idStr = tokens[0]
        label = tokens[1]
        labels[idStr] = label
    f.close()
     
    f = open(metadata, "r")
    fout=file(output,"w")
    for line0 in f:
        try:
            line = decode(line0, "gbk", 'ignore')
        except:
            continue
        tokens = line.split("\t")
        idStr = tokens[0]
        if(idStr in labels.keys()):
            toWrite = labels[idStr] + "\t" + line
            fout.write(codecs.encode(toWrite, "gbk", 'ignore'))
    f.close()
    fout.close()
    
if __name__ == '__main__':
    print "start."
#     s = loadLabels("data/train_valid/traindataset2zgb", "gbk")
#     filterDocuments("data/train_valid/split", s, "gbk")
    mergeLabelData("data/train_valid/split", "data/test/res/sorted", "data/test/testres")
    print "All finished!"