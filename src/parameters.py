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

fs = 44100
framesize_t = 0.025  # in second
hopsize_t = 0.010

framesize = int(round(framesize_t * fs))
hopsize = int(round(hopsize_t * fs))

highFrequencyBound = fs/2 if fs/2<11000 else 11000

varin = {}
# parameters of viterbi
varin['delta_mode'] = 'proportion'
varin['delta']      = 0.35

def config_select(config):
    if config[0] == 1 and config[1] == 0:
        model_name = 'single_lstm'
    elif config[0] == 1 and config[1] == 1:
        model_name = 'single_lstm_single_dense'
    elif config[0] == 2 and config[1] == 0:
        model_name = 'two_lstm'
    elif config[0] == 2 and config[1] == 1:
        model_name = 'two_lstm_single_dense'
    elif config[0] == 2 and config[1] == 2:
        model_name = 'two_lstm_two_dense'
    elif config[0] == 3 and config[1] == 0:
        model_name = 'three_lstm'
    elif config[0] == 3 and config[1] == 1:
        model_name = 'three_lstm_single_dense'
    elif config[0] == 3 and config[1] == 2:
        model_name = 'three_lstm_two_dense'
    elif config[0] == 3 and config[1] == 3:
        model_name = 'three_lstm_three_dense'
    else:
        raise ValueError

    return model_name