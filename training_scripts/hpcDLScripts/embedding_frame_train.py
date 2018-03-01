"""
script to train the frame level embedding
"""

import sys, os

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preparation import load_data_embedding_to_frame_level
from models import train_model_validation
import pickle
import h5py
import atexit

def exit_handler(filenames):
    """delete files when exits"""
    for fn in filenames:
        if os.path.isfile(fn):
            os.remove(fn)

def main():
    # filename_feature = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/feature_phn_embedding_train.pkl'
    # filename_list_key = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/list_key.pkl'
    # filename_scaler = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/scaler_phn_embedding.pkl'
    # filename_train_validation_set = '/scratch/phoneEmbeddingModelsTraining/feature_frame.h5'
    # filename_labels_train_validation_set = '/scratch/phoneEmbeddingModelsTraining/labels.pkl'
    #
    # path_model = '/homedtic/rgong/phoneEmbeddingModelsTraining/out/'

    filename_feature = '/Users/gong/Documents/MTG document/dataset/phoneEmbedding/feature_phn_embedding_train.pkl'
    filename_list_key = '/Users/gong/Documents/MTG document/dataset/phoneEmbedding/list_key.pkl'
    filename_scaler = '/Users/gong/Documents/MTG document/dataset/phoneEmbedding/scaler_phn_embedding.pkl'

    filename_train_validation_set = '/Users/gong/Documents/pycharmProjects/phoneticSimilarity/temp/feature_frame.h5'
    filename_labels_train_validation_set = '/Users/gong/Documents/pycharmProjects/phoneticSimilarity/temp/labels.pkl'

    # path_model = '/Users/gong/Documents/pycharmProjects/phoneticSimilarity/temp/'
    path_mode = '../../temp'

    model_name = 'wide_frame_level_emb_teacher_student'
    input_dim = (80, 15)

    # feature, label, scaler loading
    list_feature = pickle.load(open(filename_feature, 'rb'))
    list_key = pickle.load(open(filename_list_key, 'rb'))
    scaler = pickle.load(open(filename_scaler, 'rb'))

    array_feature_replicated, array_labels = load_data_embedding_to_frame_level(list_feature=list_feature,
                                                                                list_key=list_key,
                                                                                scaler=scaler)
    # write feature and label to files
    h5f = h5py.File(filename_train_validation_set, 'w')
    h5f.create_dataset('feature_all', data=array_feature_replicated)
    h5f.close()

    pickle.dump(array_labels, open(filename_labels_train_validation_set, 'wb'))

    for ii in range(5):
        file_path_model = os.path.join(path_model, model_name + '_' + str(ii) + '.h5')
        file_path_log = os.path.join(path_model, 'log', model_name + '_' + str(ii) + '.csv')

        train_model_validation(filename_train_validation_set=filename_train_validation_set,
                               filename_labels_train_validation_set=filename_labels_train_validation_set,
                               filter_density=4,
                               dropout=0.32,
                               input_shape=input_dim,
                               file_path_model=file_path_model,
                               filename_log=file_path_log,
                               channel=1)

    # clean the feature file
    atexit.register(exit_handler, filenames=[filename_train_validation_set, filename_labels_train_validation_set])


if __name__ == '__main__':
    main()