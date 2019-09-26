
import numpy as np
import os
from configparser import ConfigParser
from generator import AugmentedImageSequence
from models.keras import ModelFactory
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from utility import get_sample_counts


def main(fold,gender_train,gender_test):
    # parser config
    config_file = 'config_multiclas.ini'
    cp = ConfigParser()
    cp.read(config_file)

    root_output_dir= cp["DEFAULT"].get("output_dir") 

    # default config
    output_dir= root_output_dir+gender_train+'/run_'+str(fold)+'/output/'

    base_model_name = cp["DEFAULT"].get("base_model_name")
    class_names = cp["DEFAULT"].get("class_names").split(",")
    image_source_dir = cp["DEFAULT"].get("image_source_dir")

    # train config
    image_dimension = cp["TRAIN"].getint("image_dimension")

    # test config
    batch_size = cp["TEST"].getint("batch_size")
    test_steps = cp["TEST"].get("test_steps")
    use_best_weights = cp["TEST"].getboolean("use_best_weights")

    # parse weights file path
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    weights_path = os.path.join(output_dir, output_weights_name)
    best_weights_path = os.path.join(output_dir, f"best_{output_weights_name}")

    # get test sample count
    test_counts, _ = get_sample_counts(root_output_dir+gender_test+'/run_'+str(fold),"test", class_names)

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
        dataset_csv_file=os.path.join(root_output_dir+gender_test+'/run_'+str(fold), "test.csv"),
     
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

    y_pred_dir = output_dir + "y_pred_run_" + str(fold)+"_train_"+gender_train+"_test_"+gender_test+ ".csv"
    y_true_dir = output_dir + "y_true__run_" + str(fold)+"_train_"+gender_train+"_test_"+gender_test+ ".csv"


    np.savetxt(y_pred_dir, y_hat, delimiter=",")
    np.savetxt(y_true_dir, y, delimiter=",")

if __name__ == "__main__":

	genders_train=['M']
	genders_test= ['M','F']

	for fold in range (20):
		for gender_train in genders_train:
			for gender_test in genders_test:
				main(fold=fold,gender_train=gender_train,gender_test=gender_test)
	

         
