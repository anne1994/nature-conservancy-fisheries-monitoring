#! /usr/bin/python2.7

import matplotlib
matplotlib.use('Agg')
import os.path
import tensorflow as tf
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt

log_dir = "/data/tmp/inception_v3_svm_log"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

with tf.Session() as sess:
    model_filename = "./tensorflow_inception_graph.pb"
    with gfile.FastGFile(model_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name="")

    # Stage A
    filter0_tensor = sess.graph.get_tensor_by_name("conv:0")
    
    # Stage B
    filter2_tensor = sess.graph.get_tensor_by_name("conv_2:0")
    
    # Stage C
    filter_mix2_tensor = sess.graph.get_tensor_by_name("mixed_2/join:0")
    
    # Stagc D
    filter_mix7_tensor = sess.graph.get_tensor_by_name("mixed_7/join:0")

    # bottleneck feature
    feature_tensor = sess.graph.get_tensor_by_name("pool_3:0")

    # query image that going to feed into the inception network
    image_data = gfile.FastGFile("/data/models/slim/train/images/SHARKimg_01881.jpg", "rb").read()
    
    # feed in the query image and retrieve tensor value at different stages
    filter0, filter2, filter_mix2, filter_mix7, feature = sess.run(
        [filter0_tensor, filter2_tensor, filter_mix2_tensor, filter_mix7_tensor, feature_tensor], {
            "DecodeJpeg/contents:0": image_data
        })

    filter0 = filter0[0]
    filter2 = filter2[0]
    filter_mix2 = filter_mix2[0]
    filter_mix7 = filter_mix7[0]
    feature = feature[0][0][0]
   

    # plot the bottleneck feature
    print(feature.shape)
    plt.bar(range(2048), feature, width=3, color="blue", linewidth=0)
    plt.xlim(xmax=2048)
    plt.show()
    plt.savefig("/data/models/slim/bottleneck_feature_shark.png")
    print("bottleneck feature done")


    #ploting feature at stage A, B, C and D
    
    fig, axes = plt.subplots(4, 8, sharex=True, sharey=True)
    for y in range(4):
        for x in range(8):
            axes[y][x].axis("off")
            axes[y][x].imshow(filter0[:, :, y * 8 + x], cmap=plt.cm.gray)
            axes[y][x].set_adjustable('box-forced')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.show()
    plt.savefig("/data/models/slim/stageA_shark.png")
    print("stage A done")
    fig, axes = plt.subplots(4, 8, sharex=True, sharey=True)
    for y in range(4):
        for x in range(8):
            axes[y][x].axis("off")
            axes[y][x].imshow(filter2[:, :, y * 8 + x], cmap=plt.cm.gray)
            axes[y][x].set_adjustable('box-forced')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.show()
    plt.savefig("/data/models/slim/stageB_shark.png")
    print("stage B done")
    fig, axes = plt.subplots(16, 18, sharex=True, sharey=True)
    for y in range(16):
        for x in range(18):
            axes[y][x].axis("off")
            axes[y][x].imshow(filter_mix2[:, :, y * 18 + x], cmap=plt.cm.gray)
            axes[y][x].set_adjustable('box-forced')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.show()
    plt.savefig("/data/models/slim/stageC_shark.png")
    print("stage C done")
    fig, axes = plt.subplots(24, 32, sharex=True, sharey=True)
    for y in range(24):
        for x in range(32):
            axes[y][x].axis("off")
            axes[y][x].imshow(filter_mix7[:, :, y * 32 + x], cmap=plt.cm.gray)
            axes[y][x].set_adjustable('box-forced')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.show()
    plt.savefig("/data/models/slim/stageD_shark.png")
    print("stage D done")
    
    # plot the bottleneck feature
    #print(feature.shape)
    #plt.bar(range(2048), feature, width=3, color="blue", linewidth=0)
    #plt.xlim(xmax=2048)
    #plt.show()
    #plt.savefig("/data/models/slim/bottleneck_feature_glovemarks.png")
    #print("bottleneck feature done")
