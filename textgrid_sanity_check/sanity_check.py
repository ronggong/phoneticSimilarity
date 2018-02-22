"""
check the annotation errors in textgrid
"""

from src.train_test_filenames import getTestRecordingsJoint
from src.textgridParser import syllableTextgridExtraction
from src.filepath import *
from src.phonemeMap import dic_pho_map
import os

def s_check(textgrid_path,
            recordings,
            parentTierName,
            childTierName):

    for artist_path, recording in recordings:
        nestedLists, _, _   \
            = syllableTextgridExtraction(textgrid_path=textgrid_path,
                                         recording=os.path.join(artist_path,recording),
                                         tier0=parentTierName,
                                         tier1=childTierName)

        for ii, line_list in enumerate(nestedLists):
            print(artist_path, recording ,ii, len(line_list[1]))

            if childTierName=='details':
                for phn in line_list[1]:
                    try:
                        key = dic_pho_map[phn[2]]
                    except:
                        print(artist_path, ii, recording, phn[2])
                        raise KeyError

if __name__ == '__main__':
    # check line contains a reasonable syllable or phoneme number
    valPrimarySchool, testPrimarySchool \
        = getTestRecordingsJoint()


    s_check(textgrid_path=primarySchool_textgrid_path,
            parentTierName='line',
            childTierName='dianSilence',
            recordings=testPrimarySchool)

    s_check(textgrid_path=primarySchool_textgrid_path,
            parentTierName='line',
            childTierName='details',
            recordings=testPrimarySchool)



