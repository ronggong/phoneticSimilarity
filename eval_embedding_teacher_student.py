import csv
import os
import numpy as np
from eval_embedding import ground_truth_matrix
from eval_embedding import eval_embeddings
from data_preparation import load_data_embedding_teacher_student
from parameters import config_select
from models_RNN import model_select
from keras.models import load_model
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


def embedding_classifier_ap(filename_feature_teacher,
                            filename_list_key_teacher,
                            filename_feature_student,
                            filename_list_key_student,
                            filename_scaler):
    """calculate average precision of classifier embedding"""

    list_feature_flatten_val, label_integer_val, le, scaler = \
        load_data_embedding_teacher_student(filename_feature_teacher=filename_feature_teacher,
                                            filename_list_key_teacher=filename_list_key_teacher,
                                            filename_feature_student=filename_feature_student,
                                            filename_list_key_student=filename_list_key_student,
                                            filename_scaler=filename_scaler)

    path_model = '/Users/gong/Documents/pycharmProjects/phoneticSimilarity/models/phone_embedding_classifier'
    path_eval = '/Users/gong/Documents/pycharmProjects/phoneticSimilarity/eval/phone_embedding_classifier'

    # configs = [[1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2], [3, 3]]
    configs = [[1, 1]]

    for config in configs:
        model_name = config_select(config)

        list_ap = []
        embedding_dim = 54

        for ii in range(5):
            filename_model = os.path.join(path_model, model_name + '_teacher_student' + '_' + str(ii) + '.h5')
            model = load_model(filepath=filename_model)
            weights = model.get_weights()

            input_shape = [1, None, 80]
            model_1_batch = model_select(config=config, input_shape=input_shape, output_shape=embedding_dim)
            model_1_batch.compile(optimizer='adam',
                                  loss='categorical_crossentropy',
                                  metrics=['accuracy'])
            model_1_batch.set_weights(weights=weights)

            embeddings = np.zeros((len(list_feature_flatten_val), embedding_dim))
            for ii_emb in range(len(list_feature_flatten_val)):
                print('calculate', ii, 'run time', ii_emb, 'embedding', len(list_feature_flatten_val), 'total')

                x_batch = np.expand_dims(scaler.transform(list_feature_flatten_val[ii_emb]), axis=0)
                embeddings[ii_emb, :] = model_1_batch.predict_on_batch(x_batch)

            # dist_mat = distance_matrix_embedding_classifier(embeddings)

            dist_mat = (2.0 - squareform(pdist(embeddings, 'cosine')))/2.0
            gt_mat = ground_truth_matrix(label_integer_val)

            np.save(file=os.path.join(path_eval, 'dist_mat_' + 'teacher_student_' + str(ii)), arr=dist_mat)

            ap = eval_embeddings(dist_mat=dist_mat, gt_mat=gt_mat)

            list_ap.append(ap)

        filename_eval = os.path.join(path_eval, model_name + '_teacher_student' + '.csv')
        with open(filename_eval, 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', )
            csvwriter.writerow([np.mean(list_ap), np.std(list_ap)])


if __name__ == '__main__':
    # teacher_student val set
    filename_feature_teacher = '/Users/gong/Documents/MTG document/dataset/phoneEmbedding/feature_phn_embedding_val_teacher.pkl'
    filename_list_key_teacher = '/Users/gong/Documents/MTG document/dataset/phoneEmbedding/list_key_teacher.pkl'
    filename_feature_student = '/Users/gong/Documents/MTG document/dataset/phoneEmbedding/feature_phn_embedding_val_student.pkl'
    filename_list_key_student = '/Users/gong/Documents/MTG document/dataset/phoneEmbedding/list_key_student.pkl'
    filename_scaler = '/Users/gong/Documents/MTG document/dataset/phoneEmbedding/scaler_phn_embedding_train_teacher_student.pkl'

    embedding_classifier_ap(filename_feature_teacher,
                            filename_list_key_teacher,
                            filename_feature_student,
                            filename_list_key_student,
                            filename_scaler)