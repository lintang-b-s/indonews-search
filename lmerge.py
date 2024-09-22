import sys
n =2
"""
algoritma buat dynamic indexing
https://nlp.stanford.edu/IR-book/html/htmledition/dynamic-indexing-1.html

"""
import os 
import re 

def lMergeAddToken(indexes, z0, token):
    
    z0.append(token)# merge Z0 dengan token
    if len(z0 )==n:
        
        print("indexes already on disk: ", indexes)
        for i in range(0, sys.maxsize ):
            
            if i in indexes:
                # z[i+1] = merge(zi, Ii)
                indexes.remove(i)
                # remove inverted index file I[i] dari disk
            else:
                #I[i] = Zi
                
                indexes.add(i)
                print("i", i , " saved to disk")
                # save I[i] ke disk
                break
        z0 = []
    return z0, indexes


def logarithmicMerge():
    z0 = []
    indexes = set()
    while True:
        print("masukkan token:")  
        token = input()
        z0, indexes = lMergeAddToken(indexes, z0, token)

if __name__ == "__main__":
    logarithmicMerge()
    