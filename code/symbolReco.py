################################################################
# symbolReco.py
#
# Program that select hypothesis of segmentation
#
# Author: H. Mouchere, Dec. 2018
# Copyright (c) 2018, Harold Mouchere
################################################################
import sys
import os
import random
import itertools
import numpy as np
import torch
import sys, getopt
from convertInkmlToImg import parse_inkml,get_traces_data, getStrokesFromLG, convert_to_imgs, parseLG
from skimage.io import imsave
from torchvision.transforms import Compose, ToTensor, Normalize
from modules import SegmentSelector, AlexNet

from globals import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Instantiating the model used for predicting if a stroke combination is valid or not
model = AlexNet(101)
model.to(device)
model.load_state_dict(dict(torch.load('segmentReco.nn')))
model.eval()
img_to_tensor = Compose([
  ToTensor(),
  Normalize((0.5,), (0.5,))])


def usage():
    print ("usage: python3 symbolReco.py [-s] [-o fname][-w weigthFile] inkmlfile lgFile ")
    print ("     inkmlfile  : input inkml file name ")
    print ("     lgFile     : input LG file name")
    print ("     -o fname / --output fname : output file name (LG file)")
    print ("     -w fname / --weight fname : weight file name (nn pytorch file)")
    print ("     -s         : save hyp images")

"""
take an hypothesis (from LG = list of stroke index), select the corresponding strokes (from allTraces) and 
return the probability of being each symbol as a dictionnary {class_0 : score_0 ; ... class_i : score_i } 
Keep only the classes with a score higher than a threshold
"""

def computeClProb(alltraces, hyp, min_threshol, saveIm = False):
    # Convert image to tensor
    im = np.squeeze(np.asarray(convert_to_imgs(get_traces_data(alltraces, hyp[1]), IMG_SIZE)))
    if saveIm:
        imsave(hyp[0] + '.png', im)
    im_tensor = img_to_tensor(im)
    # Give it a "batch size" of 1
    im_tensor = im_tensor.unsqueeze(0)
    im_tensor = im_tensor.to(device)

    # create the list of possible classes
    classes = [x[0].replace('../data/symbol_recognition/','') for x in os.walk('../data/symbol_recognition/')][1:] # all subdirectories, except itself

    # Get probabilites of symbols, but keep only those who exceed threshold.
    result = {}
    with torch.no_grad():
      output = torch.nn.functional.softmax(model(im_tensor),dim=1)
    for index, val in enumerate(output[0]):
      if val.item()>min_threshol:
        result[classes[index]]=val.item()


    ## artificially simulate network output (sum(p_i) = 1)
    problist = [random.random()*random.random() for x in classes]
    sumprob = sum(problist)
    problistnorm = [p / sumprob for p in problist]
    for i,x in enumerate(classes):
        prob = problistnorm[i]
        if prob > min_threshol:
            result[x] = prob

    return result

def main():

    try:
        opts, args = getopt.getopt(sys.argv[1:], "so:w:", ["output=", "weight="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    if len(args) != 2:
        print("Not enough parameters")
        usage()
        sys.exit(2)
    inputInkml = args[0]
    inputLG = args[1]
    saveimg = False
    outputLG = ""
    weightFile = "myweight.nn"

    for o, a in opts:
        if o in ("-s"):
            saveimg = True
        elif o in ("-o", "--output"):
            outputLG = a
        elif o in ("-w", "--weight"):
            weightFile = a
        else:
            usage()
            assert False, "unhandled option"

    traces = parse_inkml(inputInkml)
    hyplist = open(inputLG, 'r').readlines()
    hyplist = parseLG(hyplist)
    output = ""
    for h in hyplist:
        # for each hypo, call the classifier and keep only selected classes (only the best or more)
        prob_dict = computeClProb(traces, h, 0.5, saveimg)
        #rewrite the new LG
        for cl, prob in prob_dict.items():
            output += "O,"+ h[0]+","+cl+","+str(prob)+","+",".join([str(s) for s in h[1]]) + "\n"
    if outputLG != "":
        with open(outputLG, "w") as text_file:
            print(output, file=text_file)
    else:
        print(output)


if __name__ == "__main__":
    # execute only if run as a script
    main()
