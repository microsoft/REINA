import sys, os, lucene, threading, time
import math
from multiprocessing import Pool
import shutil

from datetime import datetime

from org.apache.lucene import analysis, document, index, queryparser, search, store, util
from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import \
    FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions, DirectoryReader
from org.apache.lucene.store import SimpleFSDirectory, MMapDirectory
from org.apache.lucene.store import RAMDirectory
from org.apache.lucene.search.similarities import BM25Similarity, TFIDFSimilarity
import random

import json
import string
import glob
import bz2
import gzip
import sys
from tqdm import tqdm
from nltk import sent_tokenize
from nltk import word_tokenize as tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from datasets import Dataset

stops_en = set(stopwords.words('english'))
exclude = set(string.punctuation)

def remove_punc(text):
    return ''.join(ch for ch in text if ch not in exclude)

def word_tokenize(text, lowercase=True):
    words = tokenize(text)
    outputs = []
    for token in words:
        if token not in stops_en and token not in exclude:
            outputs.append( remove_punc(token) )

    return ' '.join(outputs[:600])

class MyMemLucene():

    def __init__(self):

        lucene.initVM()
        # # # lucene # # #
        self.t1 = FieldType()
        self.t1.setStored(True)
        self.t1.setTokenized(False)
        self.t1.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        self.t2 = FieldType()
        self.t2.setStored(True)
        self.t2.setTokenized(True)
        self.t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        self.t3 = FieldType()
        self.t3.setStored(True)

        self.analyzer = StandardAnalyzer()


    def built_RAM(self, data, key, value):
        self.index_directory = RAMDirectory()
        config = IndexWriterConfig( self.analyzer )
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        iwriter = IndexWriter(self.index_directory, config)

        print('Building REINA index ...')
        qbar = tqdm(total=len(data[key]))

        for instance_key, instance_value in zip(data[key], data[value]):
            doc = Document()
            doc.add(Field(key, instance_key, self.t2))
            doc.add(Field(value, instance_value, self.t2))

            try:
                iwriter.addDocument(doc)
            except:
                print(instance_value)
                continue
            qbar.update(1)
        qbar.close()
        iwriter.close()

    def retrieve_RAM(self, lines, docs_num, key, value):

        ireader = DirectoryReader.open(self.index_directory)
        isearcher = search.IndexSearcher(ireader)
        isearcher.setSimilarity(BM25Similarity())

        parser = queryparser.classic.QueryParser( key, self.analyzer)

        output_all = []
        for question in lines:
            try:
                query = parser.parse(question)
            except:
                try:
                    query = parser.parse(word_tokenize(question))
                except:
                    output_all.append(question)
                    continue

    
            hits = isearcher.search(query, max(20, docs_num) ).scoreDocs
            output = []
            for hit in hits:
                hitDoc = isearcher.doc(hit.doc)
                try:
                    if hitDoc[key] == question: continue
                    output.append( hitDoc[value] )
                    
                except:
                    continue

            instance = ' '.join( question.split(' ')[:600] )   + ' ' + ' '.join(output[:docs_num])
            output_all.append(instance)

        return output_all
        
class MultiprocessingEncoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        global mylc
        mylc = MyMemLucene()
        mylc.built_RAM( self.args['index_data'] , self.args['key'], self.args['value'] )


    def retrieve_lines(self, lines):
        output = mylc.retrieve_RAM( lines, 5, self.args['key'], self.args['value'] )
        return output


def reina_apply(raw_datasets, key, value, num_proc):
    
    index_data_list = raw_datasets['train']
    query_data_dict = {k:v for k, v in raw_datasets.items()}
    datasets_new = defaultdict(dict)

    retriever = MultiprocessingEncoder({'index_data': index_data_list, 'key': key, 'value': value})
    pool = Pool(num_proc, initializer=retriever.initializer)
    

    for set_name, query_data in query_data_dict.items():
        print(set_name)
        lines = [  k  for k in query_data[key] ]
        datasets_new[set_name][value] = [ v for v in query_data[value] ]

        encoded_lines = pool.imap(retriever.retrieve_lines, zip(*[lines]), 100)
        print('REINA start ...')
        lines_reina = []
        qbar = tqdm(total=len(query_data[key]))
        key_id = 0
        for line_id, lines_ir in enumerate(encoded_lines):
            for line in lines_ir:
                lines_reina.append(line)
                key_id += 1
            qbar.update(len(lines_ir))
            
        datasets_new[set_name][key] = lines_reina

        qbar.close()
        datasets_new[set_name] = Dataset.from_dict(datasets_new[set_name])
    return datasets_new

def reina(raw_datasets, key, value, use_cache, num_proc=10):

    import torch
    import pickle
    
    reina_path = os.getenv("HF_DATASETS_CACHE",os.path.join(os.path.expanduser('~'), '.cache/huggingface/datasets/'))
    reina_path = os.path.join(reina_path, 'reina')
    reina_dataset_path = os.path.join(reina_path, 'reina_dataset.pkl')
    
    if torch.cuda.current_device() == 0:
        print('REINA path for cache: ' + reina_dataset_path)
        print('Please remove it if data modified!')

    if not use_cache and torch.cuda.current_device() == 0:
        datasets_new = reina_apply(raw_datasets, key, value, num_proc)

        if not os.path.isdir(reina_path):
            os.makedirs(reina_path)
        with open(reina_dataset_path, 'wb') as fpw:
            pickle.dump(datasets_new, fpw)
     
    torch.distributed.barrier()
    with open(reina_dataset_path, 'rb') as fpr:
        datasets_new = pickle.load(fpr)

    return datasets_new

def reina_offline(data_name, data_path, key, value, num_proc):
    from datasets import load_dataset
    datasets = load_dataset(data_name)
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    print(datasets)

    datasets_new = reina_apply(datasets, key, value, num_proc)
    for set_name in ['validation', 'test', 'train']:
        if set_name not in datasets_new: continue

        print('REINA for ' + set_name)
        with open(os.path.join(data_path, set_name + '.json'), 'w', encoding='utf8') as fpw:
            data_num = len(datasets_new[set_name][key])
            for data_id, data in enumerate(datasets_new[set_name]):
                fpw.write(json.dumps({key: data[key], value: data[value]}) + '\n')
            fpw.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataname', type=str, default='xsum',
                        help='dataset name, such as xsum')
    parser.add_argument('--key_column', type=str, default='document',
                        help='REINA key')
    parser.add_argument('--value_column', type=str, default='summary',
                        help='REINA value')
    parser.add_argument('--reina_workers', type=int, default=10,
                        help='REINA workers')

    args = parser.parse_args()

    reina_path = os.getenv("HF_DATASETS_CACHE",os.path.join(os.path.expanduser('~'), '.cache/huggingface/datasets/'))
    reina_path = os.path.join(reina_path, 'reina', args.dataname)
    
    reina_offline(args.dataname, reina_path, args.key_column, args.value_column, args.reina_workers)

