#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
map_english_phone39_token = {
    'aa': 0, 'ae': 1, 'ah': 2, 'aw': 3, 'ay': 4, 'b': 5, 'ch': 6, 'd': 7, 'dh': 8, 'dx': 9, 'eh': 10, 'er': 11,
    'ey': 12, 'f': 13, 'g': 14, 'hh': 15, 'ih': 16, 'iy': 17, 'jh': 18, 'k': 19, 'l': 20, 'm': 21, 'n': 22, 'ng': 23,
    'ow': 24, 'oy': 25, 'p': 26, 'r': 27, 's': 28, 'sh': 29, 't': 30, 'th': 31, 'uh': 32, 'uw': 33, 'v': 34, 'w': 35,
    'y': 36, 'z': 37, 'sil': 38
}

map_english_phone48_phone66 = {
    'iy': ['iy'], 'ih': ['ih'], 'eh': ['eh'], 'ae': ['ae'], 'ix': ['ix'], 'ax': ['ax'], 'ah': ['ah', 'ax-h'],
    'uw': ['uw', 'ux'], 'uh': ['uh'], 'ao': ['ao'], 'aa': ['aa'], 'ey': ['ey'], 'ay': ['ay'], 'oy': ['oy'], 's': ['s'],
    'aw': ['aw'], 'ow': ['ow'], 'l': ['l'], 'el': ['el'], 'r': ['r'], 'y': ['y'], 'w': ['w'], 'er': ['er', 'axr'],
    'm': ['m', 'em'], 'n': ['n', 'nx'], 'en': ['en'], 'ng': ['ng', 'eng'], 'ch': ['ch'], 'jh': ['jh'], 'dh': ['dh'],
    'b': ['b'], 'd': ['d'], 'dx': ['dx'], 'g': ['g'], 'p': ['p'], 't': ['t', 'q'], 'k': ['k'], 'z': ['z'], 'zh': ['zh'],
    'v': ['v'], 'f': ['f'], 'th': ['th'], 'sh': ['sh'], 'hh': ['hh', 'hv'], 'sil': ['sil', 'h#', '#h', 'pau'],
    'cl': ['cl', 'pcl', 'tcl', 'kcl', 'qcl'], 'vcl': ['vcl', 'bcl', 'dcl', 'gcl'], 'epi': ['epi']
}

map_english_phone39_phone48 = {
    'w': ['w'], 'ch': ['ch'], 's': ['s'], 'sh': ['sh', 'zh'], 'n': ['n', 'en'], 'z': ['z'], 'ih': ['ih', 'ix'],
    'uh': ['uh'], 'ey': ['ey'], 'ay': ['ay'], 'aw': ['aw'], 'eh': ['eh'], 'r': ['r'], 'v': ['v'], 'l': ['el', 'l'],
    'aa': ['aa', 'ao'], 'jh': ['jh'], 'g': ['g'], 'm': ['m'], 'dh': ['dh'], 'k': ['k'], 'oy': ['oy'], 'uw': ['uw'],
    'ow': ['ow'], 'y': ['y'], 'ae': ['ae'], 'f': ['f'], 'b': ['b'], 't': ['t'], 'sil': ['epi', 'cl', 'vcl', 'sil'],
    'ah': ['ah', 'ax'], 'ng': ['ng'], 'th': ['th'], 'd': ['d'], 'er': ['er'], 'iy': ['iy'], 'p': ['p'], 'dx': ['dx'],
    'hh': ['hh']
}

map_english_viseme13_phone39 = {  #视素与音素的对应
    'v1': ['aa', 'ah', 'aw', 'er', 'oy'], 'sil': ['sil'], 'p': ['p', 'b', 'm'],
    'v3': ['ae', 'eh', 'ey', 'ay', 'y'], 'sh': ['sh', 'ch', 'jh'], 'l': ['l', 'r'],
    'v2': ['uw', 'uh', 'ow', 'w'], 'g': ['g', 'ng', 'k', 'hh'], 'z': ['z', 's'], 't': ['t', 'd', 'n', 'dx'],
    'th': ['th', 'dh'], 'f': ['f', 'v'], 'v4': ['ih', 'iy']
}

map_english_viseme13_token = {  #视素
    'f': 0, 'g': 1, 'l': 2, 'p': 3, 'sh': 4, 't': 5, 'th': 6, 'v1': 7, 'v2': 8, 'v3': 9, 'v4': 10, 'z': 11, 'sil': 12
}

map_english_place10_phone39 = {     #10种唇形的发音位置和39种音素的对应
    'coronal': ['d', 'l', 'n', 's', 't', 'z', 'dx'], 'high': ['ch', 'ih', 'iy', 'jh', 'sh', 'uh', 'uw', 'y'],
    'dental': ['dh', 'th'], 'glottal': ['hh'], 'labial': ['b', 'f', 'm', 'p', 'v', 'w'],
    'low': ['aa', 'ae', 'aw', 'ay', 'oy'], 'mid': ['ah', 'eh', 'ey', 'ow'], 'retroflex': ['er', 'r'],
    'velar': ['g', 'k', 'ng'], 'silence': ['sil']
}

map_english_place10_token = {  #英语10种发音位置
    'coronal': 0, 'dental': 1, 'glottal': 2, 'high': 3, 'labial': 4, 'low': 5, 'mid': 6, 'retroflex': 7, 'velar': 8,
    'silence': 9
}

map_english_manner6_phone39 = {
    'vowel': ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'oy', 'ow', 'uh', 'uw', 'er'],
    'fricative': ['jh', 'ch', 's', 'sh', 'z', 'f', 'th', 'v', 'dh', 'hh'], 'nasal': ['m', 'n', 'ng'],
    'stop': ['b', 'd', 'g', 'p', 't', 'k', 'dx'], 'approximant': ['w', 'y', 'l', 'r'], 'silence': ['sil']
}

map_mandarin_phone179_token = {
    '_a': 0, '_e': 1, '_i': 2, '_o': 3, '_u': 4, '_v': 5, 'a1': 6, 'a2': 7, 'a3': 8, 'a4': 9, 'ai1': 10, 'ai2': 11,
    'ai3': 12, 'ai4': 13, 'an1': 14, 'an2': 15, 'an3': 16, 'an4': 17, 'ang1': 18, 'ang2': 19, 'ang3': 20, 'ang4': 21,
    'ao1': 22, 'ao2': 23, 'ao3': 24, 'ao4': 25, 'b': 26, 'c': 27, 'ch': 28, 'd': 29, 'e1': 30, 'e2': 31, 'e3': 32,
    'e4': 33, 'ei1': 34, 'ei2': 35, 'ei3': 36, 'ei4': 37, 'en1': 38, 'en2': 39, 'en3': 40, 'en4': 41, 'eng1': 42,
    'eng2': 43, 'eng3': 44, 'eng4': 45, 'er2': 46, 'er3': 47, 'er4': 48, 'f': 49, 'g': 50, 'h': 51, 'i1': 52, 'i2': 53,
    'i3': 54, 'i4': 55, 'ia1': 56, 'ia2': 57, 'ia3': 58, 'ia4': 59, 'ian1': 60, 'ian2': 61, 'ian3': 62, 'ian4': 63,
    'iang1': 64, 'iang2': 65, 'iang3': 66, 'iang4': 67, 'iao1': 68, 'iao2': 69, 'iao3': 70, 'iao4': 71, 'ie1': 72,
    'ie2': 73, 'ie3': 74, 'ie4': 75, 'ii1': 76, 'ii2': 77, 'ii3': 78, 'ii4': 79, 'iii1': 80, 'iii2': 81, 'iii3': 82,
    'iii4': 83, 'iiii4': 84, 'in1': 85, 'in2': 86, 'in3': 87, 'in4': 88, 'ing1': 89, 'ing2': 90, 'ing3': 91, 'ing4': 92,
    'iong1': 93, 'iong2': 94, 'iong3': 95, 'iong4': 96, 'iou1': 97, 'iou2': 98, 'iou3': 99, 'iou4': 100, 'j': 101,
    'k': 102, 'l': 103, 'm': 104, 'n': 105, 'o1': 106, 'o2': 107, 'o3': 108, 'o4': 109, 'ong1': 110, 'ong2': 111,
    'ong3': 112, 'ong4': 113, 'ou1': 114, 'ou2': 115, 'ou3': 116, 'ou4': 117, 'p': 118, 'q': 119, 'r': 120, 's': 121,
    'sh': 122, 'sp': 123, 't': 124, 'u1': 125, 'u2': 126, 'u3': 127, 'u4': 128, 'ua1': 129, 'ua2': 130, 'ua3': 131,
    'ua4': 132, 'uai1': 133, 'uai2': 134, 'uai3': 135, 'uai4': 136, 'uan1': 137, 'uan2': 138, 'uan3': 139, 'uan4': 140,
    'uang1': 141, 'uang2': 142, 'uang3': 143, 'uang4': 144, 'uei1': 145, 'uei2': 146, 'uei3': 147, 'uei4': 148,
    'uen1': 149, 'uen2': 150, 'uen3': 151, 'uen4': 152, 'ueng1': 153, 'ueng4': 154, 'uo1': 155, 'uo2': 156, 'uo3': 157,
    'uo4': 158, 'v1': 159, 'v2': 160, 'v3': 161, 'v4': 162, 'van1': 163, 'van2': 164, 'van3': 165, 'van4': 166,
    've1': 167, 've2': 168, 've3': 169, 've4': 170, 'vn1': 171, 'vn2': 172, 'vn3': 173, 'vn4': 174, 'x': 175, 'z': 176,
    'zh': 177, 'sil': 178
}

map_mandarin_phone61_phone179 = {
    'a': ['_a', 'a1', 'a2', 'a3', 'a4'], 'e': ['_e', 'e1', 'e2', 'e3', 'e4'], 'i': ['_i', 'i1', 'i2', 'i3', 'i4'],
    'o': ['_o', 'o1', 'o2', 'o3', 'o4'], 'u': ['_u', 'u1', 'u2', 'u3', 'u4'], 'v': ['_v','v1', 'v2', 'v3', 'v4'],
    'ai': ['ai1', 'ai2', 'ai3', 'ai4'], 'an': ['an1', 'an2', 'an3', 'an4'], 'ang': ['ang1', 'ang2', 'ang3', 'ang4'],
    'ao': ['ao1', 'ao2', 'ao3', 'ao4'], 'b': ['b'], 'c': ['c'], 'ch': ['ch'], 'd': ['d'], 'k': ['k'], 'l': ['l'],
    'ei': ['ei1', 'ei2', 'ei3', 'ei4'], 'en': ['en1', 'en2', 'en3', 'en4'], 'eng': ['eng1', 'eng2', 'eng3', 'eng4'],
    'er': ['er2', 'er3', 'er4'], 'f': ['f'], 'g': ['g'], 'h': ['h'], 'ia': ['ia1', 'ia2', 'ia3', 'ia4'], 'm': ['m'],
    'ian': ['ian1', 'ian2', 'ian3', 'ian4'], 'iang': ['iang1', 'iang2', 'iang3','iang4'], 'n': ['n'], 'j': ['j'],
    'iao': ['iao1', 'iao2', 'iao3', 'iao4'], 'ie': ['ie1', 'ie2', 'ie3', 'ie4'], 'ii': ['ii1', 'ii2', 'ii3', 'ii4'],
    'iii': ['iii1', 'iii2', 'iii3', 'iii4'], 'iiii': ['iiii4'], 'in': ['in1', 'in2', 'in3', 'in4'], 'p': ['p'],
    'ing': ['ing1', 'ing2','ing3', 'ing4'], 'iong': ['iong1', 'iong2', 'iong3', 'iong4'], 'q': ['q'], 'r': ['r'],
    'iou': ['iou1', 'iou2', 'iou3', 'iou4'], 'ong': ['ong1', 'ong2', 'ong3', 'ong4'], 'x': ['x'], 'z': ['z'],
    'ou': ['ou1', 'ou2', 'ou3', 'ou4'], 's': ['s'], 'sh': ['sh'], 'sil': ['sil', 'sp'],
    'ua': ['ua1', 'ua2', 'ua3', 'ua4'], 't': ['t'], 'uai': ['uai1', 'uai2', 'uai3', 'uai4'],
    'uan': ['uan1', 'uan2', 'uan3', 'uan4'], 'uang': ['uang1', 'uang2', 'uang3', 'uang4'], 'zh': ['zh'],
    'uei': ['uei1', 'uei2', 'uei3', 'uei4'], 'uen': ['uen1', 'uen2', 'uen3', 'uen4'], 'ueng': ['ueng1', 'ueng4'],
    'uo': ['uo1', 'uo2', 'uo3', 'uo4'], 'van': ['van1', 'van2', 'van3', 'van4'], 've': ['ve1', 've2', 've3', 've4'],
    'vn': ['vn1', 'vn2', 'vn3', 'vn4']
}

map_mandarin_phone61_token = {
    'a': 0, 'ai': 1, 'an': 2, 'ang': 3, 'ao': 4, 'b': 5, 'c': 6, 'ch': 7, 'd': 8, 'e': 9, 'ei': 10, 'en': 11, 'eng': 12,
    'er': 13, 'f': 14, 'g': 15, 'h': 16, 'i': 17, 'ia': 18, 'ian': 19, 'iang': 20, 'iao': 21, 'ie': 22, 'ii': 23,
    'iii': 24, 'iiii': 25, 'in': 26, 'ing': 27, 'iong': 28, 'iou': 29, 'j': 30, 'k': 31, 'l': 32, 'm': 33, 'n': 34,
    'o': 35, 'ong': 36, 'ou': 37, 'p': 38, 'q': 39, 'r': 40, 's': 41, 'sh': 42, 't': 43, 'u': 44, 'ua': 45, 'uai': 46,
    'uan': 47, 'uang': 48, 'uei': 49, 'uen': 50, 'ueng': 51, 'uo': 52, 'v': 53, 'van': 54, 've': 55, 'vn': 56, 'x': 57,
    'z': 58, 'zh': 59, 'sil': 60
}

map_mandarin_phone32_phone61 = {  #phone61包括声母和韵母
    'b': ['b'], 'p': ['p'], 'm': ['m'], 'f': ['f'], 'd': ['d'], 'l': ['l'],
    'n': ['n', 'an', 'en', 'ian', 'in', 'uan', 'uen', 'van', 'vn'], 't': ['t'],
    'c': ['c'], 's': ['s'], 'z': ['z'], 'i1': ['ii'], 'zh': ['zh'], 'ch': ['ch'], 'sh': ['sh'],
    'r': ['r'], 'er': ['er'], 'i2': ['iii', 'iiii'], 'j': ['j'], 'q': ['q'], 'x': ['x'],
    'a': ['a', 'ai', 'ao', 'ia', 'iao', 'ua', 'uai'], 'o': ['o', 'ou', 'uo'], 'e': ['e', 'ei', 'ie', 've'],
    'i': ['i', 'iou'], 'u': ['u', 'uei'], 'v': ['v'], 'g': ['g'], 'h': ['h'], 'k': ['k'],
    'ng': ['ang', 'eng', 'iang', 'ing', 'iong', 'ong', 'uang', 'ueng'], 'sil': ['sil']
}

map_mandarin_phone32_token = {  #汉语普通话的32个音素 其中10个元音，22个辅音
    'a': 0, 'b': 1, 'c': 2, 'ch': 3, 'd': 4, 'e': 5, 'er': 6, 'f': 7, 'g': 8, 'h': 9, 'i': 10, 'i1': 11, 'i2': 12,
    'j': 13, 'k': 14, 'l': 15, 'm': 16, 'n': 17, 'ng': 18, 'o': 19, 'p': 20, 'q': 21, 'r': 22, 's': 23, 'sh': 24,
    't': 25, 'u': 26, 'v': 27, 'x': 28, 'z': 29, 'zh': 30, 'sil': 31
}

map_mandarin_place8_phone32 = {
    'bilabial': ['b', 'p', 'm'], 'labiodental': ['f'],  'alveolar': ['d', 'l', 'n', 't'],
    'dental': ['c', 's', 'z', 'i1'], 'retroflex': ['zh', 'ch', 'sh', 'r', 'er', 'i2'],
    'palatal': ['j', 'q', 'x', 'a', 'o', 'e', 'i', 'u', 'v'],  'velar': ['g', 'h', 'k', 'ng'], 'silence': ['sil']
}

map_mandarin_place8_token = {  #普通话的发音位置
    'alveolar': 0, 'bilabial': 1, 'dental': 2, 'labiodental': 3, 'palatal': 4, 'retroflex': 5, 'velar': 6, 'silence': 7
}

map_mandarin_place10_token = {
    'coronal': 0, 'spn': 1, 'glottal': 2, 'high': 3, 'labial': 4, 'low': 5, 'mid': 6, 'retroflex': 7, 'velar': 8,
    'sil': 9
}

map_mandarin_pinyin61_token = {  
    'a': 0, 'ai': 1, 'an': 2, 'ang': 3, 'ao': 4, 'b': 5, 'c': 6, 'ch': 7, 'd': 8, 'e': 9, 'ei': 10, 'en': 11, 'eng': 12,
    'er': 13, 'f': 14, 'g': 15, 'h': 16, 'ia': 17, 'iang': 18, 'ian': 19, 'iao': 20, 'ie': 21, 'i': 22, 'ing': 23,
    'in': 24, 'iong': 25, 'iu': 26, 'j': 27, 'k': 28, 'l': 29, 'm': 30, 'n': 31, 'o': 32, 'ong': 33, 'ou': 34,
    'p': 35, 'q': 36, 'r': 37, 'sh': 38, 's': 39, 't': 40, 'uai': 41, 'uang': 42, 'uan': 43, 'ua': 44, 'ui': 45, 'un': 46,
    'uo': 47, 'u': 48, 'ue': 49, 've': 50, 'v': 51, 'vn': 52, 'van': 53, 'w': 54, 'x': 55, 'y': 56, 'zh': 57,
    'z': 58, 'sil': 59, 'spn': 60
}

# avse challenge
map_english_phone41_token = {
    'aa': 0, 'ah': 1, 'ao': 2, 'aw': 3, 'er': 4, 'hh': 5, 'oy': 6, 'ow': 7, 'uh': 8, 'uw': 9, 'ae': 10, 'ay': 11,
    'eh': 12, 'ey': 13, 'ih': 14, 'iy': 15, 'l': 16, 'r': 17, 'y': 18, 's': 19, 'z': 20, 'd': 21, 'n': 22, 't': 23,
    'ch': 24, 'jh': 25, 'sh': 26, 'zh': 27, 'b': 28, 'm': 29, 'p': 30, 'dh': 31, 'th': 32, 'f': 33, 'v': 34, 'g': 35,
    'k': 36, 'ng': 37, 'w': 38, 'sil': 39, 'spn': 40
}

map_english_place11_token = {
    'coronal': 0, 'spn': 1, 'glottal': 2, 'high': 3, 'labial': 4, 'low': 5, 'mid': 6, 'retroflex': 7, 'velar': 8, 'dental': 9,
    'sil': 10
}

map_english_phone41_place11 = {
    'aa': 'low', 'ah': 'mid', 'ao': 'low', 'aw': 'low', 'er': 'retroflex', 'hh': 'glottal', 'oy': 'low', 'ow': 'mid', 'uh': 'high', 
    'uw': 'high', 'ae': 'low', 'ay': 'low', 'eh': 'mid', 'ey': 'mid', 'ih': 'high', 'iy': 'high', 'l': 'coronal', 'r': 'retroflex',
    'y': 'high', 's': 'coronal', 'z': 'coronal', 'd': 'coronal', 'n': 'coronal', 't': 'coronal', 'ch': 'high', 'jh': 'high', 
    'sh': 'high', 'zh': 'high', 'b': 'labial', 'm': 'labial', 'p': 'labial', 'dh': 'dental', 'th': 'dental', 'f': 'labial', 
    'v': 'labial', 'g': 'velar', 'k': 'velar', 'ng': 'velar', 'w': 'labial', 'sil': 'sil', 'spn': 'spn'
}


# shift2second = {
#     'train': {0: 224256.72, 1: 430037.60000000073, 2: 310072.2080059999, 3: 146039.48},
#     'test': {0: 27014.76, 1: 83473.79999999977, 2: 30017.279999999104, 3: 13806.560000000005}
# }
#
# map_anterior_phone39 = ['b', 'd', 'dh', 'f', 'l', 'm', 'n', 'p', 's', 't', 'th', 'v', 'z', 'w', 'dx']
#
# back2phone = ['ay', 'aa', 'ah', 'ao', 'aw', 'ow', 'oy', 'uh', 'uw', 'g', 'k']
#
# continuant2phone = [
#     'aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'dh', 'eh', 'er', 'r', 'ey', 'l', 'f', 'ih', 'iy', 'oy', 'ow', 's', 'sh',
#     'th', 'uh', 'uw', 'v', 'w', 'y', 'z']
#
# round2phone = ['aw', 'ow', 'uw', 'ao', 'uh', 'v', 'y', 'oy', 'r', 'w']
#
# tense2phone = ['aa', 'ae', 'aw', 'ay', 'ey', 'iy', 'ow', 'oy', 'uw', 'ch', 's', 'sh', 'f', 'th', 'p', 't', 'k', 'hh']
#
# voiced2phone = [
#     'aa', 'ae', 'ah', 'aw', 'ay', 'b', 'd', 'dh', 'dx', 'eh', 'er', 'ey', 'g', 'ih', 'iy', 'jh', 'l', 'm', 'n', 'ng',
#     'ow', 'oy', 'r', 'uh', 'uw', 'v', 'w', 'y', 'z']


