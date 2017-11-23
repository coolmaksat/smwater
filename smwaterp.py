#!/usr/bin/env python

import numpy as np
import sys
import os
from multiprocessing import Pool
import time

def main():
    start_time = time.time()
    try:
        args = sys.argv
        if len(args) != 2:
            raise Exception('Please provide input fasta file')
        filename = args[1]
        f = open(filename, 'r')
        global sequences
        proteins, sequences = read_fasta(f)
        f.close()
        n = len(sequences)
        index = list()
        for i in range(n):
            for j in range(i + 1, n):
                index.append(n * i + j)
        p = Pool(16)
        ident = p.map(score, index)
        for ind, pident in zip(index, ident):
            i = ind // n
            j = ind % n
            print('%s\t%s\%.6f', (proteins[i], proteins[j], pident))
        print('End', time.time() - start_time)
    except Exception as e:
        print(e)


def score(ind):
    n = len(sequences)
    i = ind // n
    j = ind % n
    s = sequences[i]
    t = sequences[j]
    return smith_waterman(s, t)
    
def smith_waterman(s, t):
    n = len(s)
    m = len(t)
    h = np.zeros((n + 1, m + 1), dtype=np.int32)
    max_score = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            maxk = h[i - 1, j] - 2
            for k in range(2, i + 1):
                maxk = max(maxk, h[i - k, j] - 2 * k)
            maxl = h[i, j - 1] - 2
            for l in range(2, j + 1):
                maxl = max(maxl, h[i, j - l] - 2 * l)
            h[i, j] = max(0, max(maxl, maxk))
            if s[i - 1] == t[j - 1]:
                h[i, j] = max(h[i, j], h[i - 1, j - 1] + 3)
            else:
                max(h[i, j], h[i - 1, j - 1] - 3)
            max_score = max(max_score, h[i, j])
    return max_score / (max(m, n) * 3)
            

def read_fasta(lines):
    seqs = list()
    info = list()
    seq = ''
    inf = ''
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if seq != '':
                seqs.append(seq)
                info.append(inf)
                seq = ''
            inf = line[1:]
        else:
            seq += line
    seqs.append(seq)
    info.append(inf)
    return info, seqs


if __name__ == '__main__':
    main()
