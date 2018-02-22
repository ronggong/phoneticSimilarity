'''
 * Copyright (C) 2017  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of jingjuSingingPhraseMatching
 *
 * pypYIN is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 *
 * If you have any problem about this python version code, please contact: Rong Gong
 * rong.gong@upf.edu
 *
 *
 * If you want to refer this code, please use this article:
 *
'''

from os.path import join

###################################
###### set this dataset path ######
###################################

# dataset_path = '/Users/gong/Documents/MTG document/Jingju arias/jingju_a_cappella_singing_dataset/'
# dataset_path = '/path/to/your/jingju_a_cappella_singing_dataset/'


cnn_file_name = 'gop_model'

primarySchool_dataset_root_path     = '/Users/gong/Documents/MTG document/Jingju arias/primary_school_recording'
primarySchool_wav_path = join(primarySchool_dataset_root_path, 'wav')
primarySchool_textgrid_path = join(primarySchool_dataset_root_path, 'textgrid')

# nacta 2017 dataset part 2
nacta2017_dataset_root_path     = '/Users/gong/Documents/MTG document/Jingju arias/jingju_a_cappella_singing_dataset_extended_nacta2017'
nacta2017_wav_path = join(nacta2017_dataset_root_path, 'wav')
nacta2017_textgrid_path = join(nacta2017_dataset_root_path, 'textgridDetails')

# nacta dataset part 1
nacta_dataset_root_path     = '/Users/gong/Documents/MTG document/Jingju arias/jingju_a_cappella_singing_dataset'
nacta_wav_path = join(nacta_dataset_root_path, 'wav')
nacta_textgrid_path = join(nacta_dataset_root_path, 'textgrid')

# acoustic model training dataset path
data_path_GOP_model = '/Users/gong/Documents/MTG document/dataset/GOPModels'