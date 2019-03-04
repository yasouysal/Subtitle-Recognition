#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk
import numpy as np
import math
from collections import Counter

class Vectorizer:
    def __init__(self, min_word_length=3, max_df=1.0, min_df=0.0):
        self.min_word_length = min_word_length
        self.max_df=max_df
        self.min_df=min_df
        self.term_df_dict = {}
        self.seen_words_dict = {}
        self.vocabulary = []

    def fit(self, raw_documents):
        """Generates vocabulary for feature extraction. 
        Ignores words shorter than min_word_length and document frequency
        not between max_df and min_df.

        :param raw_documents: list of string for creating vocabulary
        :return: None
        """
        self.document_count = len(raw_documents)
        # TODO: Implement this method
        
        for document in raw_documents:
            for word in set(document.split()):
                if len(word) >= self.min_word_length:
                    if word in self.seen_words_dict:
                        self.seen_words_dict[word] = self.seen_words_dict[word] + 1
                    else:
                        self.seen_words_dict.update({word:1})

        for key in self.seen_words_dict:
            ratio = self.seen_words_dict[key] / float(self.document_count)
            if ratio <= self.max_df and ratio >= self.min_df:
                self.vocabulary.append(key)
                self.term_df_dict.update({key:(self.seen_words_dict[key])})
        

    def _transform(self, raw_document, method):
        """Creates a feature vector for given raw_document according to vocabulary.

        :param raw_document: string
        :param method: one of count, existance, tf-idf
        :return: numpy array as feature vector
        """
        # TODO: Implement this method
        if method == "existance":
            vector_to_append = []
            for element in self.vocabulary:
                if element in raw_document:
                    vector_to_append.append(1)
                else:
                    vector_to_append.append(0)
            return np.array(vector_to_append)

        elif method == "count":
            vector_to_append = []
            for element in self.vocabulary:
                vector_to_append.append(raw_document.count(element))
            return np.array(vector_to_append)

        elif method == "tf-idf":
            vector_to_append = []
            for element in self.vocabulary:
                tf_of_element = raw_document.count(element)
                idf_of_element = math.log( (1 + self.document_count) / float(1+ self.term_df_dict[element]) ) + 1
                tf_idf_value = tf_of_element * idf_of_element
                vector_to_append.append(tf_idf_value)

            # print tf_of_element

            norm_for_vector = np.linalg.norm(vector_to_append)
            
            # print norm_for_vector
            
            for i in range (len(vector_to_append)):
                if norm_for_vector != 0:
                    vector_to_append[i] = vector_to_append[i] / float(norm_for_vector)

            return np.array(vector_to_append)


    def transform(self, raw_documents, method="tf-idf"):
        """For each document in raw_documents calls _transform and returns array of arrays.

        :param raw_documents: list of string
        :param method: one of count, existance, tf-idf
        :return: numpy array of feature-vectors
        """
        # TODO: Implement this method
        self.document_count = len(raw_documents)

        vector_of_vectors = []
        for document in raw_documents:
            vector_of_vectors.append(self._transform(document, method))
        return np.array(vector_of_vectors)
        

    def fit_transform(self, raw_documents, method="tf-idf"):
        """Calls fit and transform methods respectively.

        :param raw_documents: list of string
        :param method: one of count, existance, tf-idf
        :return: numpy array of feature-vectors
        """
        # TODO: Implement this method
        self.fit(raw_documents)
        return np.array(self.transform(raw_documents, method))

    def get_feature_names(self):
        """Returns vocabulary.

        :return: list of string
        """
        try:
            self.vocabulary
        except AttributeError:
            print "Please first fit the model."
            return []
        return self.vocabulary

    def get_term_dfs(self):
        """Returns number of occurances for each term in the vocabulary in sorted.

        :return: array of tuples
        """
        return sorted(self.term_df_dict.iteritems(), key=lambda (k, v): (v, k), reverse=True)

if __name__=="__main__":
    v = Vectorizer(min_df=0, max_df=1)
    contents = [
     "this is the first document",
     "this document is the second document",
     "and this is the third one",
     "is this the first document",
 ]
    v.fit(contents)
    print v.get_feature_names()
    existance_vector = v.transform(contents, method="existance")        
    print existance_vector
    count_vector = v.transform(contents, method="count")        
    print count_vector
    tf_idf_vector = v.transform(contents, method="tf-idf")
    print tf_idf_vector
