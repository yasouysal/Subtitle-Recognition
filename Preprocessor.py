#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk
import os
from nltk.corpus import stopwords
import codecs
import errno
import string
import re

class Preprocessor:
    def __init__(self, dataset_directory="Dataset", processed_dataset_directory= "ProcessedDataset"):
        self.dataset_directory = dataset_directory
        self.processed_dataset_directory=processed_dataset_directory
        nltk.download("stopwords")
        nltk.download("punkt")
        self.stop_words = set(stopwords.words('english'))

    def _remove_puncs_numbers_stop_words(self, tokens):
        """Remove punctuations in the words, words including numbers and words in the stop_words list.

        :param tokens: list of string
        :return: list of string with cleaned version
        """
        # TODO: Implement this method
        # stop = stopwords.words('english') + list(string.punctuation) + [u"’"] + [u"..."]
        # clearedlist = []
        # for token in tokens:
        #     if token not in stop:
        #         # if token.isalnum():
        #         #     clearedlist.append(token)
        #         clearedlist += [token for token in re.sub(r'[.,!?]', '', token.lower()).split() if not re.search(r'\d', token)]
        # # return [token for token in tokens if token not in stop]
        # return clearedlist

        no_number_list = []
        for token in tokens:
            token = re.sub(r'[^\w\s]', '',token.lower())
            if token not in self.stop_words:
                # token = re.sub(r'[^\w\s]', '',token.lower())
                no_number_list += [token for token in re.sub(r'[.,!?]', '', token.lower()).split() if not re.search(r'\d', token)]

        return no_number_list

    def _tokenize(self, sentence):
        """Tokenizes given string.

        :param sentence: string to tokenize
        :return: list of string with tokens
        """
        # TODO: Implement this method
        tokens = nltk.word_tokenize(sentence)
        return tokens

    def _stem(self, tokens):
        """Stems the tokens with nltk SnowballStemmer

        :param tokens: list of string
        :return: list of string with words stems
        """
        # TODO: Implement this method
        stemmer = nltk.stem.SnowballStemmer("english")
        i = 0
        for i in range (len(tokens)):
            tokens[i] = stemmer.stem(tokens[i])
        return tokens

    def preprocess_document(self, document):
        """Calls methods _tokenize, _remove_puncs_numbers_stop_words and _stem respectively.

        :param document: string to preprocess
        :return: string with processed version
        """
        # TODO: Implement this method
        # tokens = self._tokenize(document.decode("utf-8").lower())
        tokens = self._tokenize(document.lower())
        no_punc_tokens = self._remove_puncs_numbers_stop_words(tokens)
        list_to_assemble = self._stem(no_punc_tokens)
        return " ".join(list_to_assemble)


    def preprocess(self):
        """Walks through the given directory and calls preprocess_document method. The output is
        persisted into processed_dataset_directory by keeping directory structure.

        :return: None
        """
        for root, dirs, files in os.walk(self.dataset_directory):
            if os.path.basename(root) != self.dataset_directory:
                print "Processing", root, "directory."
                dest_dir = self.processed_dataset_directory+"/"+root.lstrip(self.dataset_directory+"/")
                if not os.path.exists(dest_dir):
                    try:
                        os.makedirs(dest_dir)
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise
                for file in files:
                    file_path = root + "/" + file
                    with codecs.open(file_path, "r", "ISO-8859-1") as f:
                        data = f.read().replace("\n", " ")
                    processed_data = self.preprocess_document(data)
                    output_file_path = dest_dir + "/" + file
                    with codecs.open(output_file_path, "w", "ISO-8859-1") as o:
                        o.write(processed_data)

if __name__=="__main__":
    text =  """ Greetings, ananinami66
                shall I sit or stand?
                - Tell us.
                - Tell us.
                I'll tell. We bought the goods
                from Black Faik.
                We reloaded the truck
                in Karabuk.
                I was driving the truck
                till Adana.
                - What are you talking about?
                - And you?!
                You've abducted me,
                you'll do the talking.
                I'm confused anyway.
                - Aggressive.
                - Aggressive.
                Yeah, aggressive.
                Is that it?"""
    p = Preprocessor()
    print p.preprocess_document(text)
    
    # p = Preprocessor()
    # p.preprocess()
