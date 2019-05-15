# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:49:14 2019

@author: Siddhant
"""
from segment import main as main1
from testrun import main as main2
import os


def execute(path):
    #path = os.path.join(os.getcwd(),img_name)
    #print("path is: " + path)
    img = main1(path)
    #print("img_name:" + img_name)
    output = main2(img)
    return output

def main():
    param="image.jpg"
    path = os.path.join(os.getcwd(),'uploads',param)
    if(os.path.isfile(path)):
        return execute(path)
    else:
        print("error reading file.")

'''          
if __name__ == '__main__':
    
    param = '' #Enter name of file or folder not whole path.
    main(param)
'''     
