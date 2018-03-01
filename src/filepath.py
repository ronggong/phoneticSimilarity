import os

root_path = os.path.join(os.path.dirname(__file__),'..')

primarySchool_dataset_root_path = '/Users/gong/Documents/MTG document/Jingju arias/primary_school_recording'

primarySchool_wav_path = os.path.join(primarySchool_dataset_root_path, 'wav')
primarySchool_textgrid_path = os.path.join(primarySchool_dataset_root_path, 'textgrid')

# human rating
primarySchool_human_rating_path = os.path.join(primarySchool_dataset_root_path, 'human_rating')

# gop models
cnn_file_name = 'gop'
kerasScaler_path = os.path.join(root_path, 'models', 'gop_models', 'scaler_' + cnn_file_name + '.pkl')
kerasModels_path = os.path.join(root_path, 'models', 'gop_models', cnn_file_name + '_model' + '.h5')

# classifier models
cnn_file_emb_cla_name = 'single_lstm_single_dense_0'
kerasScaler_emb_cla_path = os.path.join(root_path, 'models', 'phone_embedding_classifier', 'scaler_phn_embedding_cla.pkl')
kerasModels_emb_cla_path = os.path.join(root_path, 'models', 'phone_embedding_classifier', cnn_file_emb_cla_name + '.h5')

# frame level models
cnn_file_emb_frame_level_name = 'wide_frame_level_emb_0'
kerasScaler_emb_frame_level_path = os.path.join(root_path, 'models', 'phone_embedding_classifier', 'scaler_phn_embedding_cla.pkl')
kerasModels_emb_frame_level_path = os.path.join(root_path, 'models', 'phoneme_embedding_frame_level', cnn_file_emb_frame_level_name + '.h5')