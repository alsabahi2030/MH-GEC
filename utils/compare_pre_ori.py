from utils.helpers import read_parallel_lines,write_lines, read_lines
def read_3parallel_lines(fn1, fn2, fn3):
    lines1 = read_lines(fn1, skip_strip=True)
    lines2 = read_lines(fn2, skip_strip=True)
    lines3 = read_lines(fn3, skip_strip=True)
    assert len(lines1) == len(lines2) == len(lines3)
    out_lines1, out_lines2, out_lines3 = [], [], []
    for line1, line2, line3 in zip(lines1, lines2, lines3):
        if not line1.strip() or not line2.strip() or not  line3.strip():
            continue
        else:
            out_lines1.append(line1)
            out_lines2.append(line2)
            out_lines3.append(line3)

    return out_lines1, out_lines2, out_lines3



gold_file ="/data_local/src/gector-master/data/wi.dev.cor"
ori_file ="/data_local/src/gector-master/data/wi.dev.ori"
predict_file ="/data_local/src/gector-master/outputs/outputs_ensemble_1_valid_errant2/xlnet.suffixvalidpiedaev3mc2a05.c2013trainfinetunec2013winlfcfinetune2.ep0.wi.dev.mp0.0.cf0.0x200.cor"
correct_file = "/data_local/src/gector-master/data/bea_dev_stats/wi.dev.y"
incorrect_file = "/data_local/src/gector-master/data/bea_dev_stats/wi.dev.x"

pre_data, target_data, ori_data = read_3parallel_lines(predict_file, gold_file,ori_file)
correct_pre = []
correct_tgt = []
correct_ori = []
incorrect_ori = []
incorrect_pre = []
incorrect_tgt = []
correct_pairs = []
incorrect_pairs = []
for l1,l2, l3 in zip(pre_data,target_data, ori_data):
    if l1 == l2:
        correct_pairs.append((l1,l2, l3))
        correct_pre.append(l1)
        correct_tgt.append(l2)
        correct_ori.append(l3)

    else:
        incorrect_pairs.append((l1,l2,l3))
        incorrect_pre.append(l1)
        incorrect_tgt.append(l2)
        incorrect_ori.append(l3)

#write_lines(correct_file, correct_pairs, mode='w')
#write_lines(incorrect_file, incorrect_pairs, mode='w')
write_lines(f"{correct_file}.cor.pre", correct_pre, mode='w')
write_lines(f"{correct_file}.cor.tgt", correct_tgt, mode='w')
write_lines(f"{correct_file}.cor.ori", correct_ori, mode='w')

write_lines(f"{incorrect_file}.incor.pre", incorrect_pre, mode='w')
write_lines(f"{incorrect_file}.incor.tgt", incorrect_tgt, mode='w')
write_lines(f"{incorrect_file}.incor.ori", incorrect_ori, mode='w')
