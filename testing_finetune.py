
import numpy as np
import os
from configparser import ConfigParser
from generator import AugmentedImageSequence
from models.keras import ModelFactory
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from utility import get_sample_counts


def main(fold,gender_train,gender_test):
    # parser config
    config_file = 'config_file.ini'
    cp = ConfigParser()
    cp.read(config_file)

    from keras import backend as K
    import tensorflow as tf
    import keras

    K.tensorflow_backend._get_available_gpus()
    config = tf.ConfigProto( device_count = {'GPU': 1} ) 
    sess = tf.Session(config=config) 
    print(sess)
    keras.backend.set_session(sess)

    root_output_dir= cp["DEFAULT"].get("output_dir") 

    for finetune_name in ['','_finetune_100', '_finetune_500', '_finetune_1000', '_finetune_2500', '_finetune_5000', '_finetune_10000', '_finetune_20000']:

        # default config 
        print(root_output_dir,gender_train)   
        output_dir= root_output_dir + gender_train+'/Fold_'+str(fold)+'/output'+finetune_name+'/'

        base_model_name = cp["DEFAULT"].get("base_model_name")
        class_names = cp["DEFAULT"].get("class_names").split(",")
        image_source_dir = cp["DEFAULT"].get("image_source_dir")

        # train config
        image_dimension = cp["TRAIN"].getint("image_dimension")

        # test config
        batch_size = cp["FINETUNE_TEST"].getint("batch_size")
        test_steps = cp["FINETUNE_TEST"].get("test_steps")
        use_best_weights = cp["FINETUNE_TEST"].getboolean("use_best_weights")

        # parse weights file path
        output_weights_name = cp["FINETUNE"].get("output_weights_name")
        weights_path = os.path.join(output_dir, output_weights_name)
        best_weights_path = os.path.join(output_dir, f"best_{output_weights_name}")

        # get test sample count
        test_counts, _ = get_sample_counts(root_output_dir+gender_train+'/Fold_'+str(fold),str(gender_test), class_names)

        # compute steps
        if test_steps == "auto":
            test_steps = int(test_counts / batch_size)
        else:
            try:
                test_steps = int(test_steps)
            except ValueError:
                raise ValueError(f"""
                    test_steps: {test_steps} is invalid,
                    please use 'auto' or integer.
                    """)
        print(f"** test_steps: {test_steps} **")

        print("** load model **")
        if use_best_weights:
            print("** use best weights **")
            model_weights_path = best_weights_path
        else:
            print("** use last weights **")
            model_weights_path = weights_path
        model_factory = ModelFactory()
        model = model_factory.get_model(
            class_names,
            model_name=base_model_name,
            use_base_weights=False,
            weights_path=model_weights_path)

        print("** load test generator **")
        test_sequence = AugmentedImageSequence(
            dataset_csv_file=os.path.join(root_output_dir+gender_train+'/Fold_'+str(fold), str(gender_test)+".csv"),
        
            class_names=class_names,
            source_image_dir=image_source_dir,
            batch_size=batch_size,
            target_size=(image_dimension, image_dimension),
            augmenter=None,
            steps=test_steps,
            shuffle_on_epoch_end=False,
        )

        print("** make prediction **")

        y_hat = model.predict_generator(test_sequence, verbose=1)
        y = test_sequence.get_y_true()

        y_pred_dir = output_dir + "y_pred_run_" + str(fold)+"_train"+gender_train+"_"+gender_test+ ".csv"
        y_true_dir = output_dir + "y_true_run_" + str(fold)+"_train"+gender_train+"_"+gender_test+ ".csv"


        np.savetxt(y_pred_dir, y_hat, delimiter=",")
        np.savetxt(y_true_dir, y, delimiter=",")

if __name__ == "__main__":

	genders_train=['0%_female_images','100%_female_images']
	genders_test= ['test_female','test_males']
	n_splits=20


	for fold in range (n_splits):
		for gender_train in genders_train:
			for gender_test in genders_test:
				main(fold=fold,gender_train=gender_train,gender_test=gender_test)
	

         
