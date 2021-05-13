import numpy
import os
import argparse

def tes_gpu(modeles):

    if modeles=="tensorflow":
        print("Now test tensorflow-gpu is avaliable?")
        import tensorflow as tf
        if tf.test.gpu_device_name():
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        else:
            print("Please install GPU version of TF")

    elif modeles=="pytorch":
        print("Now test pytorch-gpu is avaliable?")
        import torch
        flag = torch.cuda.is_available()
        print(flag)
        ngpu = 1
        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    if modeles=="all":
        print("Now test tensorflow-gpu is avaliable?")
        import tensorflow as tf
        if tf.test.gpu_device_name():
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        else:
            print("Please install GPU version of TF")

        print("Now test pytorch-gpu is avaliable?")
        import torch
        flag = torch.cuda.is_available()
        print(flag)
        ngpu = 1
        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")



def main():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument(
        "--test",
        # action="store_true",
        type=str,
        required=True,
        )
    #
    args = parser.parse_args()
    # print(args)
    model=args.test
    tes_gpu(model)

if __name__=="__main__":
    main()