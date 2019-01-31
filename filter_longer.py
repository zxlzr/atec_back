#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/5/16
import random


def filter_longer(infile,outfile):
	with open(outfile, 'w') as f_out:
		for line in open(infile):
			line2=line
			line1 = line.strip().split('\t')

			if len(line1[1].decode('utf-8'))<=15 and len(line1[2].decode('utf-8'))<=15:
				#print line
			#print len(s1)<34
				f_out.write(line2)


if __name__ == '__main__':
    filter_longer('data/train_all.csv','data/atec_nlp_sim_train_15.csv')

