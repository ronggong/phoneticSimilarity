def getTestRecordingsJoint():
    # test recording for syllable and phoneme joint estimation

    # dataset to tune the parameters
    valPrimarySchool = [['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'student01'],
                         ['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'student02_first_half'],
                         ['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'student02'],
                         ['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'student03'],
                         ['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'student04'],
                         ['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'student05'],
                         ['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'student06'],

                        ['20171211SongRuoXuan/daxp-Fei_shi_wo-Hua_tian_cuo-dxjky', 'student01'],
                        ['20171211SongRuoXuan/daxp-Fei_shi_wo-Hua_tian_cuo-dxjky', 'student02'],
                        ['20171211SongRuoXuan/daxp-Fei_shi_wo-Hua_tian_cuo-dxjky', 'student03'],
                        ['20171211SongRuoXuan/daxp-Fei_shi_wo-Hua_tian_cuo-dxjky', 'student04'],

                        # ['20171217TianHao/lsxp-Wei_guo_jia-Hong_yang_dong-sizhu', 'student_01'],
                         # ['20171217TianHao/lsxp-Wei_guo_jia-Hong_yang_dong-sizhu', 'student_02'],
                         # ['20171217TianHao/lsxp-Wei_guo_jia-Hong_yang_dong-sizhu', 'student_03'],
                         # ['20171217TianHao/lsxp-Wei_guo_jia-Hong_yang_dong-sizhu', 'student_04'],
                         # ['20171217TianHao/lsxp-Wei_guo_jia-Hong_yang_dong-sizhu', 'student_05']
                         ]

    testPrimarySchool = [
                         ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky','student01'],
                         ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky','student02'],
                         ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky','student03'],
                         ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky','student04'],
                         ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky','student05'],
                         ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky','student06'],
                         ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky','student07'],
                         ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky','student08'],

                         ['20171211SongRuoXuan/daxp-Wo_jia_di-Hong_deng_ji-dxjky', 'student01'],
                        ['20171211SongRuoXuan/daxp-Wo_jia_di-Hong_deng_ji-dxjky', 'student02'],


                            # ['20171217TianHao/lsxp-Jiang_shen_er-San_jia_dian-sizhu', 'student_01'],
                            # ['20171217TianHao/lsxp-Jiang_shen_er-San_jia_dian-sizhu', 'student_02'],
                            # ['20171217TianHao/lsxp-Jiang_shen_er-San_jia_dian-sizhu', 'student_03_1'],
                            # ['20171217TianHao/lsxp-Jiang_shen_er-San_jia_dian-sizhu', 'student_03_2'],
                            # ['20171217TianHao/lsxp-Jiang_shen_er-San_jia_dian-sizhu', 'student_04_mentougou']

                            ]

    return valPrimarySchool, testPrimarySchool


def getTeacherRecordings():
    """
    recordings for training the GOP (goodness of pronunciation) models
    only containing professional singers
    :return:
    """
    trainNacta2017 = [['20170327LiaoJiaNi', 'lseh-Niang_zi_bu-Sou_gu_jiu-nacta'],  # yes, pro
                      ['20170327LiaoJiaNi', 'lsxp-Yi_ma_li-Wu_jia_po-nacta'],  # pro
                      ['20170418TianHao', 'lseh-Tan_yang_jia-Hong_yang_dong-nacta']]  # yes # pro

    trainNacta = [['danAll', 'dafeh-Bi_yun_tian-Xi_xiang_ji01-qm'],
                  ['danAll', 'danbz-Bei_jiu_chan-Chun_gui_men01-qm'],
                  ['danAll', 'danbz-Kan_dai_wang-Ba_wang_bie_ji01-qm'],
                  ['danAll', 'daspd-Hai_dao_bing-Gui_fei_zui_jiu02-qm'],
                  ['danAll', 'daxp-Chun_qiu_ting-Suo_lin_nang01-qm'],
                  ['danAll', 'daxp-Jiao_Zhang_sheng-Hong_niang01-qm'],
                  ['danAll', 'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai04-qm'],
                  ['danAll', 'daxp-Zhe_cai_shi-Suo_lin_nang01-qm'],
                  ['laosheng', 'lsxp-Guo_liao_yi-Wen_zhao_guan02-qm'],
                  ['laosheng', 'lsxp-Huai_nan_wang-Huai_he_ying02-qm'],
                  ['laosheng', 'lsxp-Jiang_shen_er-San_jia_dian02-qm']]

    # only for training acoustic model
    trainSepa = [['danAll', 'shiwenhui_tingxiongyan'],
                 ['danAll', 'xixiangji_biyuntian'],
                 ['danAll', 'xixiangji_diyilai'],
                 ['danAll', 'xixiangji_luanchouduo'],
                 ['danAll', 'xixiangji_manmufeng'],
                 ['danAll', 'xixiangji_xianzhishuo'],
                 ['danAll', 'xixiangji_zhenmeijiu'],
                 ['danAll', 'yutangchun_yutangchun'],
                 ['danAll', 'zhuangyuanmei_daocishi'],
                 ['danAll', 'zhuangyuanmei_fudingkui'],
                 ['danAll', 'zhuangyuanmei_tianbofu'],
                 ['danAll', 'zhuangyuanmei_zhenzhushan'],
                 ['danAll', 'zhuangyuanmei_zinari'],
                 ['danAll', 'wangjiangting_zhijianta'],
                 ['danAll', 'wangjiangting_dushoukong']]

    # train dataset primary school
    # train dataset primary school
    trainPrimarySchool = [['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'teacher'],
                          ['20171211SongRuoXuan/daxp-Fei_shi_wo-Hua_tian_cuo-dxjky', 'teacher'],
                          ['20171211SongRuoXuan/daxp-Wo_jia_di-Hong_deng_ji-dxjky', 'teacher'],
                          ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky', 'teacher'],
                          ['20171214SongRuoXuan/daeh-Yang_yu_huan-Tai_zhen_wai_zhuan-nanluo', 'teacher'],
                          ['20171214SongRuoXuan/danbz-Kan_dai_wang-Ba_wang_bie_ji-nanluo', 'teacher'],
                          ['20171214SongRuoXuan/daspd-Hai_dao_bing-Gui_fei_zui_jiu-nanluo', 'teacher'],
                          ['20171214SongRuoXuan/daxp-Quan_jun_wang-Ba_wang_bie_ji-nanluo', 'teacher'],
                          ['20171215SongRuoXuan/daxp-Jiao_zhang_sheng-Xi_shi-qianmen', 'teacher'],
                          ['2017121215SongRuoXuan/daxp-Mu_qin_bu_ke-Feng_huan_chao-yucai_qianmen', 'teacher'],

                          ['20171217TianHao/lsxp-Wei_guo_jia-Hong_yang_dong-sizhu', 'teacher'],
                          ['20171217TianHao/lsxp-Jiang_shen_er-San_jia_dian-sizhu', 'teacher'],
                          ['20171217TianHao/lseh-Wo_men_shi-Zhi_qu-sizhu', 'teacher'],
                          ['20171217TianHao/lsxp-Lin_xing_he_ma-Hong_deng_ji-sizhu', 'teacher'],
                          ]

    return trainNacta2017, trainNacta, trainSepa, trainPrimarySchool


def getStudentRecordings():
    # dataset to tune the parameters, fixed for the submission interspeech
    valPrimarySchool = [['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'student01'],
                        ['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'student02_first_half'],
                        ['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'student02'],
                        ['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'student03'],
                        ['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'student04'],
                        ['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'student05'],
                        ['20171211SongRuoXuan/daxp_Qing_zao_qi_lai-Mai_shui-dxjky', 'student06'],

                        ['20171211SongRuoXuan/daxp-Fei_shi_wo-Hua_tian_cuo-dxjky', 'student01'],
                        ['20171211SongRuoXuan/daxp-Fei_shi_wo-Hua_tian_cuo-dxjky', 'student02'],
                        ['20171211SongRuoXuan/daxp-Fei_shi_wo-Hua_tian_cuo-dxjky', 'student03'],
                        ['20171211SongRuoXuan/daxp-Fei_shi_wo-Hua_tian_cuo-dxjky', 'student04'],

                        ['20171217TianHao/lseh-Wo_men_shi-Zhi_qu-sizhu', 'student_01'],
                        ['20171217TianHao/lseh-Wo_men_shi-Zhi_qu-sizhu', 'student_02'],

                        ['20171217TianHao/lsxp-Wei_guo_jia-Hong_yang_dong-sizhu', 'student_01'],
                        ['20171217TianHao/lsxp-Wei_guo_jia-Hong_yang_dong-sizhu', 'student_02'],
                        ['20171217TianHao/lsxp-Wei_guo_jia-Hong_yang_dong-sizhu', 'student_03'],
                        ['20171217TianHao/lsxp-Wei_guo_jia-Hong_yang_dong-sizhu', 'student_04'],
                        ['20171217TianHao/lsxp-Wei_guo_jia-Hong_yang_dong-sizhu', 'student_05'],
                        ]

    # dataset for testing, fixed for the submission interspeech
    testPrimarySchool = [
        ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky', 'student01'],
        ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky', 'student02'],
        ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky', 'student03'],
        ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky', 'student04'],
        ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky', 'student05'],
        ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky', 'student06'],
        ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky', 'student07'],
        ['20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky', 'student08'],

        ['20171211SongRuoXuan/daxp-Wo_jia_di-Hong_deng_ji-dxjky', 'student01'],
        ['20171211SongRuoXuan/daxp-Wo_jia_di-Hong_deng_ji-dxjky', 'student02'],
        ['20171211SongRuoXuan/daxp-Wo_jia_di-Hong_deng_ji-dxjky', 'student03'],
        ['20171211SongRuoXuan/daxp-Wo_jia_di-Hong_deng_ji-dxjky', 'student04'],
        ['20171211SongRuoXuan/daxp-Wo_jia_di-Hong_deng_ji-dxjky', 'student05'],
        ['20171211SongRuoXuan/daxp-Wo_jia_di-Hong_deng_ji-dxjky', 'student06'],

        ['20171217TianHao/lsxp-Jiang_shen_er-San_jia_dian-sizhu', 'student_01'],
        ['20171217TianHao/lsxp-Jiang_shen_er-San_jia_dian-sizhu', 'student_02'],
        ['20171217TianHao/lsxp-Jiang_shen_er-San_jia_dian-sizhu', 'student_03_1'],
        ['20171217TianHao/lsxp-Jiang_shen_er-San_jia_dian-sizhu', 'student_03_2'],
        ['20171217TianHao/lsxp-Jiang_shen_er-San_jia_dian-sizhu', 'student_04_mentougou'],

        ['20171217TianHao/lsxp-Lin_xing_he_ma-Hong_deng_ji-sizhu', 'student_01'],
        ['20171217TianHao/lsxp-Lin_xing_he_ma-Hong_deng_ji-sizhu', 'student_02'],
        ]
    return valPrimarySchool, testPrimarySchool


def getExtraStudentRecordings():
    """extra test dataset adult students"""
    extra_test_adult = [['20171214SongRuoXuan/daeh-Yang_yu_huan-Tai_zhen_wai_zhuan-nanluo', 'student_01'],
                        ['20171214SongRuoXuan/daeh-Yang_yu_huan-Tai_zhen_wai_zhuan-nanluo', 'student_02'],
                        ['20171214SongRuoXuan/daeh-Yang_yu_huan-Tai_zhen_wai_zhuan-nanluo', 'student_03'],
                        ['20171214SongRuoXuan/daeh-Yang_yu_huan-Tai_zhen_wai_zhuan-nanluo', 'student_04'],
                        ['20171214SongRuoXuan/daeh-Yang_yu_huan-Tai_zhen_wai_zhuan-nanluo', 'student_05'],

                        ['20171214SongRuoXuan/danbz-Kan_dai_wang-Ba_wang_bie_ji-nanluo', 'student_01'],
                        ['20171214SongRuoXuan/danbz-Kan_dai_wang-Ba_wang_bie_ji-nanluo', 'student_02'],
                        ['20171214SongRuoXuan/danbz-Kan_dai_wang-Ba_wang_bie_ji-nanluo', 'student_03'],

                        ['20171214SongRuoXuan/daspd-Hai_dao_bing-Gui_fei_zui_jiu-nanluo', 'student_01'],
                        ['20171214SongRuoXuan/daspd-Hai_dao_bing-Gui_fei_zui_jiu-nanluo', 'student_02'],

                        ['20171214SongRuoXuan/daxp-Quan_jun_wang-Ba_wang_bie_ji-nanluo', 'student_01'],
                        ['20171214SongRuoXuan/daxp-Quan_jun_wang-Ba_wang_bie_ji-nanluo', 'student_02']]

    return extra_test_adult