#!/usr/bin/env python
# coding: utf-8
# @Author: Ningyu Zhang
# @Date  : 2018/6/13
import random


def filter_longer(infile,outfile):
	with open(outfile, 'w') as f_out:
		for line in open(infile):
			line2=line
			line1 = line.strip().split('\t')

			if len(line1[1].decode('utf-8'))==0 or len(line1[2].decode('utf-8'))==0:
				print line1
				
			#print len(s1)<34
			f_out.write(line2)


if __name__ == '__main__':
    filter_longer('data/atec_nlp_sim_train.csv','data/train_filtered.csv')

