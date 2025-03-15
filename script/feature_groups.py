import sys
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from Bio import SeqIO

def get_the_same_float(feature_value_list, number=8):
    import numpy as np
    float_list = [float(x) for x in feature_value_list]
    float_to_array = np.array(float_list)
    array_float_n = np.round(float_to_array, number)
    new_feature_list = list(array_float_n)
    return new_feature_list


class PCP9:
    def __init__(self):
        pass

    def get_feature_name(self, sequence_length):
        feature_name_list = []
        p_property = ['Hydrophobicity', 'Hydrophilicity', 'Mass', 'pK1', 'pK2', 'pI', 'Rigidity', 'Flexibility', 'Irreplaceability']
        for element in p_property:
            for i in range(1, sequence_length + 1):
                feature_name_char = 'PCP9_' + element + '_pos_' + str(i)
                feature_name_list.append(feature_name_char)
        return feature_name_list

    def main(self, sequence):
        import pandas as pd
        f_PCP = pd.read_csv('nine_physicochemical_properties_of_amino_acid_Stand.txt', sep='\t')
        # print(f_PCP.loc[8])
        # print(f_PCP.loc[0]['A'])
        AA_char = 'ACDEFGHIKLMNPQRSTVWY'
        Hyfrophobicity_dict = {}
        Hydrophilicity_dict = {}
        Mass_dict = {}
        pK1_dict = {}
        pK2_dict = {}
        pl_dict = {}
        Rigidity_dict = {}
        Flexibility_dict = {}
        Irreplaceability_dict = {}
        for char in AA_char:  # 建立性质-值的字典，提取特征的时候直接进行转换就行
            Hyfrophobicity_dict[char] = f_PCP.loc[0][char]
            Hydrophilicity_dict[char] = f_PCP.loc[1][char]
            Mass_dict[char] = f_PCP.loc[2][char]
            pK1_dict[char] = f_PCP.loc[3][char]
            pK2_dict[char] = f_PCP.loc[4][char]
            pl_dict[char] = f_PCP.loc[5][char]
            Rigidity_dict[char] = f_PCP.loc[6][char]
            Flexibility_dict[char] = f_PCP.loc[7][char]
            Irreplaceability_dict[char] = f_PCP.loc[8][char]
        # print(Irreplaceability_dict)
        feature_Hyfrophobicity = []
        feature_Hydrophilicity = []
        feature_Mass = []
        feature_pK1 = []
        feature_pK2 = []
        feature_pl = []
        feature_Rigidity = []
        feature_Flexibility = []
        feature_Irreplaceability = []
        for char1 in sequence:
            if char1 != 'X':
                feature_Hyfrophobicity.append(str(Hyfrophobicity_dict[char1]))
                feature_Hydrophilicity.append(str(Hydrophilicity_dict[char1]))
                feature_Mass.append(str(Mass_dict[char1]))
                feature_pK1.append(str(pK1_dict[char1]))
                feature_pK2.append(str(pK2_dict[char1]))
                feature_pl.append(str(pl_dict[char1]))
                feature_Rigidity.append(str(Rigidity_dict[char1]))
                feature_Flexibility.append(str(Flexibility_dict[char1]))
                feature_Irreplaceability.append(str(Irreplaceability_dict[char1]))
            else:
                feature_Hyfrophobicity.append('0')
                feature_Hydrophilicity.append('0')
                feature_Mass.append('0')
                feature_pK1.append('0')
                feature_pK2.append('0')
                feature_pl.append('0')
                feature_Rigidity.append('0')
                feature_Flexibility.append('0')
                feature_Irreplaceability.append('0')
        feature_total = []
        feature_total.extend(feature_Hyfrophobicity)
        feature_total.extend(feature_Hydrophilicity)
        feature_total.extend(feature_Mass)
        feature_total.extend(feature_pK1)
        feature_total.extend(feature_pK2)
        feature_total.extend(feature_pl)
        feature_total.extend(feature_Rigidity)
        feature_total.extend(feature_Flexibility)
        feature_total.extend(feature_Irreplaceability)
        feature = get_the_same_float(feature_total)
        return feature


class PWAA_fea:  # 该种特征要取上下游等长的情况才行
    def __init__(self):
        self.AA_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    def get_feature_name(self):
        feature_name_list = []
        for element in self.AA_list:
            featureName = 'PWAA_feature_%s' % element
            feature_name_list.append(featureName)
        return feature_name_list

    def main(self, sequence):
        length_up_down = (len(sequence) - 1) / 2
        feature = []
        for aa_char in self.AA_list:
            sum_inter = 0
            if aa_char not in sequence:
                feature.append(0)
            else:
                for sequence_index, sequence_char in enumerate(sequence):
                    if sequence_char == aa_char:
                        j = sequence_index - length_up_down  # 这里10到时要改成上下游的那个L
                        sum_inter = sum_inter + (j + abs(j) / length_up_down)
                c = (1 / (length_up_down * (length_up_down + 1))) * sum_inter
                feature.append(c)
        feature_total = get_the_same_float(feature)
        return feature_total


class ARPC_fea:  # 该种特征要取上下游等长的情况才行
    def __init__(self):
        self.AA_dict = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11,
                        'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, "X": 0}

    def get_feature_name(self, sequence_length):
        feature_name_list = []
        up_down_length = (sequence_length - 1) / 2
        for i in range(0, sequence_length):
            temp_site = i - up_down_length
            featureName = 'ARPC_site_%s' % int(temp_site)
            feature_name_list.append(featureName)
        return feature_name_list

    def main(self, sequence):
        length_up_down = (len(sequence) - 1) / 2
        feature = []

        for i in range(0, len(sequence)):
            site_value = (i - length_up_down)
            acid_code = self.AA_dict[sequence[i]]
            temp_feature = site_value * acid_code
            feature.append(temp_feature)
        feature_total = get_the_same_float(feature)
        return feature_total


class K_space:  # 注意，在使用get_feature_name的时候是默认输出0-k的名称，而在计算的时候只计算对应的那种k!!!
    def __init__(self, k):
        self.k = k
        self.feature_name = []
        self.AA_list_sort = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    def get_feature_name(self):
        for element in range(self.k + 1):
            element = str(element)
            prefix = 'Kspace_%s_space_' % element
            for i1 in self.AA_list_sort:
                for j1 in self.AA_list_sort:
                    self.feature_name.append(prefix + i1 + j1)
        return self.feature_name

    def main(self, sequence):
        two_mer_name = []
        for i2 in self.AA_list_sort:
            for j2 in self.AA_list_sort:
                combination = i2 + j2
                two_mer_name.append(combination)
        two_mer_dict = {}  # 定义一个2mer组成的字典用于计数
        for item in two_mer_name:
            two_mer_dict[item] = 0
        for i, v in enumerate(sequence):
            if i + 2 + self.k > len(sequence):  # 这里的思想是：i的索引加上1等于实际的位置，再加上k+1就是与之配对的那个（后面个）AA位置，看其超出序列的总长度没有
                break
            else:
                now_k_mer = sequence[i:i + 2 + self.k:self.k + 1]  # 后边界是取不到的，所以要定义为i+k+1
                if now_k_mer in two_mer_dict:  # 这里加一个判断是因为在补X的时候会有问题出现
                    two_mer_dict[now_k_mer] += 1
        feature_vector = []  # 这里可以直接加标签，通常以[]空的放置，[0]表示负样本标签,[1]表示正样本标签
        for key, value in two_mer_dict.items():
            frequency = value / (len(sequence) - 1 - self.k)
            feature_vector.append(frequency)
        feature = get_the_same_float(feature_vector)
        return feature


class EBGW_fea:
    def __init__(self, number_sub_sequence):
        self.number_sub_sequence = number_sub_sequence

    def get_feature_name(self):
        feature_name = []
        for i in range(1, self.number_sub_sequence + 1):
            prefix = 'EBGW_%s_feature_' % i
            total_number_sub_sequence = 3 * i
            for j in range(1, total_number_sub_sequence + 1):
                featureName = prefix + str(j)
                feature_name.append(featureName)
        return feature_name

    def main(self, sequence):
        # The hydrophobic group
        c1 = 'AFGILMPVW'
        # The polar group
        c2 = 'CNQSTY'
        # The positively charged group
        c3 = 'KHR'
        # The negatively charged group
        c4 = 'DE'
        # definition of H1
        h1_bi_sequence = ''
        for element in sequence:
            if element in c1 or element in c2:
                h1_bi_sequence += str(1)
            else:
                h1_bi_sequence += str(0)

        # definition of H2
        h2_bi_sequence = ''
        for element in sequence:
            if element in c1 or element in c3:
                h2_bi_sequence += str(1)
            else:
                h2_bi_sequence += str(0)

        # definition of H3
        h3_bi_sequence = ''
        for element in sequence:
            if element in c1 or element in c4:
                h3_bi_sequence += str(1)
            else:
                h3_bi_sequence += str(0)

        feature = []
        feature_H1_list = []
        feature_H2_list = []
        feature_H3_list = []
        for i in range(1, self.number_sub_sequence + 1):
            i_length = round(i * len(sequence) / self.number_sub_sequence)  # definition length of the i-th sub-sequence
            i_h1_sub_bi_sequence = h1_bi_sequence[0:i_length]
            sum_i_h1 = i_h1_sub_bi_sequence.count('1')
            feature_h1_sub_sequence = sum_i_h1 / i_length
            feature_H1_list.append(feature_h1_sub_sequence)

            i_h2_sub_bi_sequence = h2_bi_sequence[0:i_length]
            sum_i_h2 = i_h2_sub_bi_sequence.count('1')
            feature_h2_sub_sequence = sum_i_h2 / i_length
            feature_H2_list.append(feature_h2_sub_sequence)

            i_h3_sub_bi_sequence = h3_bi_sequence[0:i_length]
            sum_i_h3 = i_h3_sub_bi_sequence.count('1')
            feature_h3_sub_sequence = sum_i_h3 / i_length
            feature_H3_list.append(feature_h3_sub_sequence)

        for element1 in feature_H1_list:
            feature.append(element1)
        for element2 in feature_H2_list:
            feature.append(element2)
        for element3 in feature_H3_list:
            feature.append(element3)
        feature_total = get_the_same_float(feature)
        return feature_total


class CTD_fea:
    def __init__(self):
        pass

# 首先进行转化:Polar(P);Neutral(N);Hydrophobicity(H);
# 转化为：Hydrophobicity
    def _transform1(self, sequence1):
        char_1 = ''
        class1 = 'RKEDQN'
        class2 = 'GASTPHY'
        class3 = 'CLVIMFW'
        for element1 in sequence1:
            if element1 in class1:
                char_1 += 'P'
            elif element1 in class2:
                char_1 += 'N'
            elif element1 in class3:
                char_1 += 'H'
            else:
                char_1 += 'X'
        return char_1

# 转化为：Normalized van der Waals volume
    def _transform2(self, sequence1):
        char_2 = ''
        class1 = 'GASTPDC'
        class2 = 'NVEQIL'
        class3 = 'MHKFRYW'
        for element2 in sequence1:
            if element2 in class1:
                char_2 += 'P'
            elif element2 in class2:
                char_2 += 'N'
            elif element2 in class3:
                char_2 += 'H'
            else:
                char_2 += 'X'
        return char_2

# 转化为：Polarity
    def _transform3(self, sequence1):
        char_3 = ''
        class1 = 'LIFWCMVY'
        class2 = 'PATGS'
        class3 = 'HQRKNED'
        for element3 in sequence1:
            if element3 in class1:
                char_3 += 'P'
            elif element3 in class2:
                char_3 += 'N'
            elif element3 in class3:
                char_3 += 'H'
            else:
                char_3 += 'X'
        return char_3

# 转化为：Polarizability
    def _transform4(self, sequence1):
        char_4 = ''
        class1 = 'GASDT'
        class2 = 'CPNVEQIL'
        class3 = 'KMHFRYW'
        for element4 in sequence1:
            if element4 in class1:
                char_4 += 'P'
            elif element4 in class2:
                char_4 += 'N'
            elif element4 in class3:
                char_4 += 'H'
            else:
                char_4 += 'X'
        return char_4

# 转化为：Charge
    def _transform5(self, sequence1):
        char_5 = ''
        class1 = 'KR'
        class2 = 'ANCQGHILMFPSTWYV'
        class3 = 'DE'
        for element5 in sequence1:
            if element5 in class1:
                char_5 += 'P'
            elif element5 in class2:
                char_5 += 'N'
            elif element5 in class3:
                char_5 += 'H'
            else:
                char_5 += 'X'
        return char_5

# 转化为：Secondary structure
    def _transform6(self, sequence1):
        char_6 = ''
        class1 = 'EALMQKRH'
        class2 = 'VIYCWFT'
        class3 = 'GNPSD'
        for element6 in sequence1:
            if element6 in class1:
                char_6 += 'P'
            elif element6 in class2:
                char_6 += 'N'
            elif element6 in class3:
                char_6 += 'H'
            else:
                char_6 += 'X'
        return char_6

# 转化为：Solvent accessibility
    def _transform7(self, sequence1):
        char_7 = ''
        class1 = 'ALFCGIVW'
        class2 = 'PKQEND'
        class3 = 'MRSTHY'
        for element7 in sequence1:
            if element7 in class1:
                char_7 += 'P'
            elif element7 in class2:
                char_7 += 'N'
            elif element7 in class3:
                char_7 += 'H'
            else:
                char_7 += 'X'
        return char_7

    def _computing_CTD(self, PNH_sequence):
        CTD_feature_vector = []
        length_sequence = len(PNH_sequence)
        # 计算C
        number_P = PNH_sequence.count('P')
        number_N = PNH_sequence.count('N')
        number_H = PNH_sequence.count('H')
        C_P = number_P / length_sequence
        C_N = number_N / length_sequence
        C_H = number_H / length_sequence
        CTD_feature_vector.append(C_P)
        CTD_feature_vector.append(C_N)
        CTD_feature_vector.append(C_H)
        # 计算T
        number_PN = PNH_sequence.count('PN')
        number_NP = PNH_sequence.count('NP')
        number_PN_NP = number_PN + number_NP
        number_PH = PNH_sequence.count('PH')
        number_HP = PNH_sequence.count('HP')
        number_PH_HP = number_PH + number_HP
        number_NH = PNH_sequence.count('NH')
        number_HN = PNH_sequence.count('HN')
        number_NH_HN = number_NH + number_HN
        T_PN_NP = number_PN_NP / (length_sequence - 1)
        T_PH_HP = number_PH_HP / (length_sequence - 1)
        T_NH_HN = number_NH_HN / (length_sequence - 1)
        CTD_feature_vector.append(T_PN_NP)
        CTD_feature_vector.append(T_PH_HP)
        CTD_feature_vector.append(T_NH_HN)
        # 计算D
        P_list = []
        N_list = []
        H_list = []
        for i, v in enumerate(PNH_sequence):
            if v == 'P':
                P_list.append(i)
            elif v == 'N':
                N_list.append(i)
            elif v == 'H':
                H_list.append(i)
        if P_list:
            first_P = (P_list[0] + 1) / length_sequence
            per25_P = (P_list[int((len(P_list) * 0.25)) - 1] + 1) / length_sequence
            per50_P = (P_list[int((len(P_list) * 0.5)) - 1] + 1) / length_sequence
            per75_P = (P_list[int((len(P_list) * 0.75)) - 1] + 1) / length_sequence
            per100_P = (P_list[-1] + 1) / length_sequence
        else:
            first_P = 0
            per25_P = 0
            per50_P = 0
            per75_P = 0
            per100_P = 0
        if N_list:
            first_N = (N_list[0] + 1) / length_sequence
            per25_N = (N_list[int((len(N_list) * 0.25)) - 1] + 1) / length_sequence
            per50_N = (N_list[int((len(N_list) * 0.5)) - 1] + 1) / length_sequence
            per75_N = (N_list[int((len(N_list) * 0.75)) - 1] + 1) / length_sequence
            per100_N = (N_list[-1] + 1) / length_sequence
        else:
            first_N = 0
            per25_N = 0
            per50_N = 0
            per75_N = 0
            per100_N = 0
        if H_list:
            first_H = (H_list[0] + 1) / length_sequence
            per25_H = (H_list[int((len(H_list) * 0.25)) - 1] + 1) / length_sequence
            per50_H = (H_list[int((len(H_list) * 0.5)) - 1] + 1) / length_sequence
            per75_H = (H_list[int((len(H_list) * 0.75)) - 1] + 1) / length_sequence
            per100_H = (H_list[-1] + 1) / length_sequence
        else:
            first_H = 0
            per25_H = 0
            per50_H = 0
            per75_H = 0
            per100_H = 0
        CTD_feature_vector.append(first_P)
        CTD_feature_vector.append(per25_P)
        CTD_feature_vector.append(per50_P)
        CTD_feature_vector.append(per75_P)
        CTD_feature_vector.append(per100_P)
        CTD_feature_vector.append(first_N)
        CTD_feature_vector.append(per25_N)
        CTD_feature_vector.append(per50_N)
        CTD_feature_vector.append(per75_N)
        CTD_feature_vector.append(per100_N)
        CTD_feature_vector.append(first_H)
        CTD_feature_vector.append(per25_H)
        CTD_feature_vector.append(per50_H)
        CTD_feature_vector.append(per75_H)
        CTD_feature_vector.append(per100_H)
        return CTD_feature_vector

    def get_feature_name(self):
        PNH_list = ['P', 'N', 'H']
        CTD_list = ['C', 'T', 'D']
        T_list = ['PN_NP', 'PH_HP', 'NH_HN']
        D_list = ['0', '25', '50', '75', '100']
        first_row_name = []
        for i in range(1, 8):
            for j in CTD_list:
                if j == 'C':
                    for k1 in PNH_list:
                        char = ('CTD_%a_' + j + '_' + k1) % i
                        first_row_name.append(char)
                elif j == 'T':
                    for k2 in T_list:
                        char = ('CTD_%a_' + j + '_' + k2) % i
                        first_row_name.append(char)
                elif j == 'D':
                    for k3 in PNH_list:
                        for l in D_list:
                            char = ('CTD_%a_' + j + '_' + k3 + '_' + l) % i
                            first_row_name.append(char)
        return first_row_name

    def main(self, sequence):
        t1 = self._transform1(sequence)
        feature1 = self._computing_CTD(t1)
        t2 = self._transform2(sequence)
        feature2 = self._computing_CTD(t2)
        t3 = self._transform3(sequence)
        feature3 = self._computing_CTD(t3)
        t4 = self._transform4(sequence)
        feature4 = self._computing_CTD(t4)
        t5 = self._transform5(sequence)
        feature5 = self._computing_CTD(t5)
        t6 = self._transform6(sequence)
        feature6 = self._computing_CTD(t6)
        t7 = self._transform7(sequence)
        feature7 = self._computing_CTD(t7)
        feature_total = feature1 + feature2 + feature3 + feature4 + feature5 + feature6 + feature7
        feature = get_the_same_float(feature_total)
        return feature


class CC_fea:
    def __init__(self):
        pass

    def get_feature_name(self):
        feature_name_list = ['Type1Xmean', 'Type1Ymean', 'Type1Zmean',
                             'Type2Xmean', 'Type2Ymean', 'Type2Zmean',
                             'Type3Xmean', 'Type3Ymean', 'Type3Zmean',
                             'Type4Xmean', 'Type4Ymean', 'Type4Zmean',
                             'AllXmean', 'AllYmean', 'AllZmean',
                             'CumXmean', 'CumYmean', 'CumZmean']
        return feature_name_list

    def main(self, sequence):
        lst = []
        seq_length = len(sequence)
        # AA_char = 'AVLIPFWMGSTCYNQKRHDE'
        F1 = sequence.count('A')
        F2 = sequence.count('V')
        F3 = sequence.count('L')
        F4 = sequence.count('I')
        F5 = sequence.count('P')
        F6 = sequence.count('F')
        F7 = sequence.count('W')
        F8 = sequence.count('M')
        F9 = sequence.count('G')
        F10 = sequence.count('S')
        F11 = sequence.count('T')
        F12 = sequence.count('C')
        F13 = sequence.count('Y')
        F14 = sequence.count('N')
        F15 = sequence.count('Q')
        F16 = sequence.count('K')
        F17 = sequence.count('R')
        F18 = sequence.count('H')
        F19 = sequence.count('D')
        F20 = sequence.count('E')
        Type1Xvalue = F1 * (-36.6794) + F2 * (-66.6159) + F3 * (-94.4033) + F4 * (-100.0599) + F5 * (-84.6835) + F6 * (
            -101.0966) + F7 * (-12.5874) + F8 * (-111.3198)
        Type1Yvalue = F1 * (58.8302) + F2 * (62.1625) + F3 * (38.8595) + F4 * (20.2361) + F5 * (29.1446) + F6 * (
            -79.3826) + F7 * (-158.3870) + F8 * (-32.9407)
        Type1Zvalue = F1 * (-55.9682) + F2 * (-73.5564) + F3 * (-82.4134) + F4 * (-82.4134) + F5 * (-72.3001) + F6 * (
            -103.7705) + F7 * (-128.2684) + F8 * (-93.7201)
        Type2Xvalue = F9 * (-24.7561) + F10 * (-43.6432) + F11 * (-25.2154) + F12 * (-52.4159) + F13 * (
            -53.3563) + F14 * (-44.2471) + F15 * (-65.6344)
        Type2Yvalue = F9 * (26.2092) + F10 * (25.3163) + F11 * (51.3147) + F12 * (-25.2563) + F13 * (-68.7011) + F14 * (
            -45.4289) + F15 * (-24.8605)
        Type2Zvalue = F9 * (65.8804) + F10 * (92.1974) + F11 * (104.4787) + F12 * (106.3209) + F13 * (
            158.9550) + F14 * (115.8828) + F15 * (128.2518)
        Type3Xvalue = F16 * (-3.6804) + F17 * (-2.1627) + F18 * (-0.6273)
        Type3Yvalue = F16 * (-1.8878) + F17 * (-4.4286) + F18 * (4.3459)
        Type3Zvalue = F16 * (-146.1415) + F17 * (-174.1303) + F18 * (-155.1379)
        Type4Xvalue = F19 * (-15.3220) + F20 * (-16.9336)
        Type4Yvalue = F19 * (43.3370) + F20 * (-47.8954)
        Type4Zvalue = F19 * (-124.9110) + F20 * (-138.0496)
        Type1Xmean = Type1Xvalue / 8
        Type1Ymean = Type1Yvalue / 8
        Type1Zmean = Type1Zvalue / 8
        Type2Xmean = Type2Xvalue / 7
        Type2Ymean = Type2Yvalue / 7
        Type2Zmean = Type2Zvalue / 7
        Type3Xmean = Type3Xvalue / 3
        Type3Ymean = Type3Yvalue / 3
        Type3Zmean = Type3Zvalue / 3
        Type4Xmean = Type4Xvalue / 2
        Type4Ymean = Type4Yvalue / 2
        Type4Zmean = Type4Zvalue / 2
        AllXmean = (Type1Xvalue + Type2Xvalue + Type3Xvalue + Type4Xvalue) / seq_length
        AllYmean = (Type1Yvalue + Type2Yvalue + Type3Yvalue + Type4Yvalue) / seq_length
        AllZmean = (Type1Zvalue + Type2Zvalue + Type3Zvalue + Type4Zvalue) / seq_length
        lst.append(Type1Xmean)
        lst.append(Type1Ymean)
        lst.append(Type1Zmean)
        lst.append(Type2Xmean)
        lst.append(Type2Ymean)
        lst.append(Type2Zmean)
        lst.append(Type3Xmean)
        lst.append(Type3Ymean)
        lst.append(Type3Zmean)
        lst.append(Type4Xmean)
        lst.append(Type4Ymean)
        lst.append(Type4Zmean)
        lst.append(AllXmean)
        lst.append(AllYmean)
        lst.append(AllZmean)
        # 接下来计算各分量的累加坐标的均值
        lst1 = []
        lst2 = []
        lst3 = []
        for i in range(0, len(sequence)):
            if sequence[i] == 'A':
                sx = '-36.6794'
                sy = '58.8302'
                sz = '-55.9682'
                lst1.append(sx)
                lst2.append(sy)
                lst3.append(sz)

            elif sequence[i] == 'V':
                sx = '-66.6159'
                sy = '62.1625'
                sz = '73.5564'
                lst1.append(sx)
                lst2.append(sy)
                lst3.append(sz)

            elif sequence[i] == 'L':
                sx = '-94.4003'
                sy = '38.8595'
                sz = '-82.4134'
                lst1.append(sx)
                lst2.append(sy)
                lst3.append(sz)

            elif sequence[i] == 'I':
                sx = '-100.5090'
                sy = '20.2361'
                sz = '-82.4134'
                lst1.append(sx)
                lst2.append(sy)
                lst3.append(sz)

            elif sequence[i] == 'P':
                sx = '-84.6835'
                sy = '29.1446'
                sz = '-72.3001'
                lst1.append(sx)
                lst2.append(sy)
                lst3.append(sz)

            elif sequence[i] == 'F':
                sx = '-101.0996'
                sy = '-79.3826'
                sz = '-103.7705'
                lst1.append(sx)
                lst2.append(sy)
                lst3.append(sz)

            elif sequence[i] == 'W':
                sx = '-12.5874'
                sy = '-158.3870'
                sz = '-128.2684'
                lst1.append(sx)
                lst2.append(sy)
                lst3.append(sz)

            elif sequence[i] == 'M':
                sx = '-111.3198'
                sy = '-32.9407'
                sz = '-93.7201'
                lst1.append(sx)
                lst2.append(sy)
                lst3.append(sz)

            elif sequence[i] == 'G':
                sx = '-24.7561'
                sy = '26.2092'
                sz = '65.8804'
                lst1.append(sx)
                lst2.append(sy)
                lst3.append(sz)

            elif sequence[i] == 'S':
                sx = '-43.6432'
                sy = '25.313163'
                sz = '92.1974'
                lst1.append(sx)
                lst2.append(sy)
                lst3.append(sz)

            elif sequence[i] == 'T':
                sx = '-25.2154'
                sy = '51.3147'
                sz = '104.3787'
                lst1.append(sx)
                lst2.append(sy)
                lst3.append(sz)

            elif sequence[i] == 'C':
                sx = '-51.4159'
                sy = '-25.2563'
                sz = '106.3209'
                lst1.append(sx)
                lst2.append(sy)
                lst3.append(sz)

            elif sequence[i] == 'Y':
                sx = '-53.3563'
                sy = '-68.7011'
                sz = '158.9550'
                lst1.append(sx)
                lst2.append(sy)
                lst3.append(sz)

            elif sequence[i] == 'N':
                sx = '-44.2471'
                sy = '-45.4280'
                sz = '115.8828'
                lst1.append(sx)
                lst2.append(sy)
                lst3.append(sz)

            elif sequence[i] == 'Q':
                sx = '-65.6344'
                sy = '-24.8605'
                sz = '128.2518'
                lst1.append(sx)
                lst2.append(sy)
                lst3.append(sz)

            elif sequence[i] == 'K':
                sx = '-3.6804'
                sy = '-1.8878'
                sz = '-146.1415'
                lst1.append(sx)
                lst2.append(sy)
                lst3.append(sz)

            elif sequence[i] == 'R':
                sx = '-2.1627'
                sy = '-4.4286'
                sz = '-174.1303'
                lst1.append(sx)
                lst2.append(sy)
                lst3.append(sz)

            elif sequence[i] == 'H':
                sx = '-0.6273'
                sy = '4.3459'
                sz = '-155.1379'
                lst1.append(sx)
                lst2.append(sy)
                lst3.append(sz)

            elif sequence[i] == 'D':
                sx = '-15.3220'
                sy = '43.3370'
                sz = '-124.9110'
                lst1.append(sx)
                lst2.append(sy)
                lst3.append(sz)

            elif sequence[i] == 'E':
                sx = '-16.9336'
                sy = '-47.8954'
                sz = '-138.0496'
                lst1.append(sx)
                lst2.append(sy)
                lst3.append(sz)
            elif sequence[i] == 'X':
                sx = '0'
                sy = '0'
                sz = '0'
                lst1.append(sx)
                lst2.append(sy)
                lst3.append(sz)

        cumulative_sum_x = 0
        cumulative_sum_y = 0
        cumulative_sum_z = 0
        for inx1, val1 in enumerate(lst1):
            cumulative_sum_x += float(val1) * (len(lst1) - inx1)
        feature_cm_X = cumulative_sum_x / len(lst1)

        for inx2, val2 in enumerate(lst2):
            cumulative_sum_y += float(val2) * (len(lst2) - inx2)
        feature_cm_Y = cumulative_sum_y / len(lst2)

        for inx3, val3 in enumerate(lst3):
            cumulative_sum_z += float(val3) * (len(lst3) - inx3)
        feature_cm_Z = cumulative_sum_z / len(lst3)

        feature_total = []
        feature_total.extend(lst)
        feature_total.append(feature_cm_X)
        feature_total.append(feature_cm_Y)
        feature_total.append(feature_cm_Z)

        feature = get_the_same_float(feature_total)
        return feature
