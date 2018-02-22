import sys, os

if sys.argv[1] == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import train_model_validation

if __name__ == '__main__':

    nlen = 15
    input_dim = (80, nlen)

    if sys.argv[1] == '0':
        filename_train_validation_set = '/datasets/MTG/projects/compmusic/jingju_datasets/GOPModels/feature_gop.h5'
        filename_labels_train_validation_set = '/datasets/MTG/projects/compmusic/jingju_datasets/GOPModels/labels_gop.pkl'
    else:
        filename_train_validation_set = '/scratch/gopModelsTraining/feature_gop.h5'
        filename_labels_train_validation_set = '/homedtic/rgong/gopModelsTraining/dataset/labels_gop.pkl'

    file_path_model = '/homedtic/rgong/gopModelsTraining/out/gop_model.h5'
    file_path_log = '/homedtic/rgong/gopModelsTraining/out/log/gop_model.csv'

    # filename_train_validation_set = '/Users/gong/Documents/MTG document/dataset/acousticModels/feature_hsmm_am.h5'
    # filename_labels_train_validation_set = '/Users/gong/Documents/MTG document/dataset/acousticModels/labels_hsmm_am.pickle.gz'
    #
    # file_path_model = '../../temp/hsmm_am_timbral.h5'
    # file_path_log = '../../temp/hsmm_am_timbral.csv'

    train_model_validation(filename_train_validation_set,
                            filename_labels_train_validation_set,
                            filter_density=4,
                            dropout=0.32,
                            input_shape=input_dim,
                            file_path_model = file_path_model,
                            filename_log = file_path_log,
                            channel=1)