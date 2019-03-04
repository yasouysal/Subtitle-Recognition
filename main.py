# coding=utf-8

from ExperimentSuite import ExperimentSuite
import tensorflow as tf
from Preprocessor import Preprocessor
from Vectorizer import Vectorizer
import OneHotEncoder
import numpy as np

DEFAULT_EPOCH = 75
DEFAULT_LAYERS = (512,)
DEFAULT_ACTIVATION = tf.nn.relu
DEFAULT_LOSS = "categorical_hinge"


if __name__ == "__main__":
    # TODO: Make your experiments here
    es = ExperimentSuite()
    # p = Preprocessor()
    # p.preprocess()

    # print "preprocess is done"
    
    # V1 = Vectorizer(3, 0.97, 0.5)
    # V1.fit(es.train_contents)

    # V2 = Vectorizer(3, 0.97, 0.25)
    # V2.fit(es.train_contents)

    V3 = Vectorizer(3, 0.97, 0.1)
    V3.fit(es.train_contents)

    # print "vectorizer-fit is done"

    # existance_vectorized_train_data_V1 = V1.transform(es.train_contents, "existance")
    # existance_vectorized_test_data_V1 = V1.transform(es.test_contents, "existance")
    # count_vectorized_train_data_V1 = V1.transform(es.train_contents, "count")
    # count_vectorized_test_data_V1 = V1.transform(es.test_contents, "count")
    # tf_idf_vectorized_train_data_V1 = V1.transform(es.train_contents, "tf-idf")
    # tf_idf_vectorized_test_data_V1 = V1.transform(es.test_contents, "tf-idf")

    # existance_vectorized_train_data_V2 = V2.transform(es.train_contents, "existance")
    # existance_vectorized_test_data_V2 = V2.transform(es.test_contents, "existance")
    # count_vectorized_train_data_V2 = V2.transform(es.train_contents, "count")
    # count_vectorized_test_data_V2 = V2.transform(es.test_contents, "count")
    # tf_idf_vectorized_train_data_V2 = V2.transform(es.train_contents, "tf-idf")
    # tf_idf_vectorized_test_data_V2 = V2.transform(es.test_contents, "tf-idf")

    # existance_vectorized_train_data_V3 = V3.transform(es.train_contents, "existance")
    # existance_vectorized_test_data_V3 = V3.transform(es.test_contents, "existance")
    # count_vectorized_train_data_V3 = V3.transform(es.train_contents, "count")
    # count_vectorized_test_data_V3 = V3.transform(es.test_contents, "count")
    tf_idf_vectorized_train_data_V3 = V3.transform(es.train_contents, "tf-idf")
    tf_idf_vectorized_test_data_V3 = V3.transform(es.test_contents, "tf-idf")

    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)

    # print "transform and callback is done"

    # result1 = es.train_model(DEFAULT_LAYERS, tbCallBack, existance_vectorized_train_data_V1, es.train_y, existance_vectorized_test_data_V1,
    # 				 es.test_y, DEFAULT_LOSS, DEFAULT_ACTIVATION, DEFAULT_EPOCH)
    # print result1

    # result2 = es.train_model(DEFAULT_LAYERS, tbCallBack, count_vectorized_train_data_V1, es.train_y, count_vectorized_test_data_V1,
    #                  es.test_y, DEFAULT_LOSS, DEFAULT_ACTIVATION, DEFAULT_EPOCH)
    # print result2

    # result3 = es.train_model(DEFAULT_LAYERS, tbCallBack, tf_idf_vectorized_train_data_V1, es.train_y, tf_idf_vectorized_test_data_V1,
    # 				 es.test_y, DEFAULT_LOSS, DEFAULT_ACTIVATION, DEFAULT_EPOCH)
    # print result3

    # result4 = es.train_model(DEFAULT_LAYERS, tbCallBack, existance_vectorized_train_data_V2, es.train_y, existance_vectorized_test_data_V2,
    # 				 es.test_y, DEFAULT_LOSS, DEFAULT_ACTIVATION, DEFAULT_EPOCH)
    # print result4

    # result5 = es.train_model(DEFAULT_LAYERS, tbCallBack, count_vectorized_train_data_V2, es.train_y, count_vectorized_test_data_V2,
    # 				 es.test_y, DEFAULT_LOSS, DEFAULT_ACTIVATION, DEFAULT_EPOCH)
    # print result5

    # result6 = es.train_model(DEFAULT_LAYERS, tbCallBack, tf_idf_vectorized_train_data_V2, es.train_y, tf_idf_vectorized_test_data_V2,
    # 				 es.test_y, DEFAULT_LOSS, DEFAULT_ACTIVATION, DEFAULT_EPOCH)
    # print result6

    # result7 = es.train_model(DEFAULT_LAYERS, tbCallBack, existance_vectorized_train_data_V3, es.train_y, existance_vectorized_test_data_V3,
    # 				 es.test_y, DEFAULT_LOSS, DEFAULT_ACTIVATION, DEFAULT_EPOCH)
    # print result7

    # result8 = es.train_model(DEFAULT_LAYERS, tbCallBack, count_vectorized_train_data_V3, es.train_y, count_vectorized_test_data_V3,
    # 				 es.test_y, DEFAULT_LOSS, DEFAULT_ACTIVATION, DEFAULT_EPOCH)
    # print result8

    # result9 = es.train_model(DEFAULT_LAYERS, tbCallBack, tf_idf_vectorized_train_data_V3, es.train_y, tf_idf_vectorized_test_data_V3,
    # 				 es.test_y, DEFAULT_LOSS, DEFAULT_ACTIVATION, DEFAULT_EPOCH)
    # print result9
    
    
    #
    #
    # bu 3lü countu farklı layerlarla test etmek icin                             

    # result2 = es.train_model((512,), tbCallBack, count_vectorized_train_data_V3, es.train_y, count_vectorized_test_data_V3,
    # 				 es.test_y, DEFAULT_LOSS, DEFAULT_ACTIVATION, DEFAULT_EPOCH)
    # print result2

    # result5 = es.train_model((512,256,), tbCallBack, count_vectorized_train_data_V3, es.train_y, count_vectorized_test_data_V3,
    # 				 es.test_y, DEFAULT_LOSS, DEFAULT_ACTIVATION, DEFAULT_EPOCH)
    # print result5

    # result8 = es.train_model((512,256,128), tbCallBack, count_vectorized_train_data_V3, es.train_y, count_vectorized_test_data_V3,
    # 				 es.test_y, DEFAULT_LOSS, DEFAULT_ACTIVATION, DEFAULT_EPOCH)
    # print result8

    #
    #
    #


    # result3 = es.train_model((512,), tbCallBack, tf_idf_vectorized_train_data_V3, es.train_y, tf_idf_vectorized_test_data_V3,
    # 				 es.test_y, DEFAULT_LOSS, DEFAULT_ACTIVATION, DEFAULT_EPOCH)
    # print result3

    # result6 = es.train_model((512,256,), tbCallBack, tf_idf_vectorized_train_data_V3, es.train_y, tf_idf_vectorized_test_data_V3,
    # 				 es.test_y, DEFAULT_LOSS, DEFAULT_ACTIVATION, DEFAULT_EPOCH)
    # print result6

    # result9 = es.train_model((512,256,128), tbCallBack, tf_idf_vectorized_train_data_V3, es.train_y, tf_idf_vectorized_test_data_V3,
    # 				 es.test_y, DEFAULT_LOSS, DEFAULT_ACTIVATION, DEFAULT_EPOCH)
    # print result9





    # result1 = es.train_model((512,), tbCallBack, tf_idf_vectorized_train_data_V3, es.train_y, tf_idf_vectorized_test_data_V3,
    # 				 es.test_y, 'categorical_hinge', tf.nn.relu, DEFAULT_EPOCH)
    # print result1

    # result2 = es.train_model((512,), tbCallBack, tf_idf_vectorized_train_data_V3, es.train_y, tf_idf_vectorized_test_data_V3,
    # 				 es.test_y, 'categorical_crossentropy', tf.nn.relu, DEFAULT_EPOCH)
    # print result2

    # result3 = es.train_model((512,), tbCallBack, tf_idf_vectorized_train_data_V3, es.train_y, tf_idf_vectorized_test_data_V3,
    # 				 es.test_y, 'categorical_hinge', tf.nn.tanh, DEFAULT_EPOCH)
    # print result3

    # result4 = es.train_model((512,), tbCallBack, tf_idf_vectorized_train_data_V3, es.train_y, tf_idf_vectorized_test_data_V3,
    # 				 es.test_y, 'categorical_crossentropy', tf.nn.tanh, DEFAULT_EPOCH)
    # print result4

    # result5 = es.train_model((512,), tbCallBack, tf_idf_vectorized_train_data_V3, es.train_y, tf_idf_vectorized_test_data_V3,
    # 				 es.test_y, 'categorical_hinge', tf.nn.sigmoid, DEFAULT_EPOCH)
    # print result5

    # result6 = es.train_model((512,), tbCallBack, tf_idf_vectorized_train_data_V3, es.train_y, tf_idf_vectorized_test_data_V3,
    # 				 es.test_y, 'categorical_crossentropy', tf.nn.sigmoid, DEFAULT_EPOCH)
    # print result6



    result = es.train_model((512,), tbCallBack, tf_idf_vectorized_train_data_V3, es.train_y, tf_idf_vectorized_test_data_V3,
    				 es.test_y, 'categorical_crossentropy', tf.nn.sigmoid, 500)
    print result

