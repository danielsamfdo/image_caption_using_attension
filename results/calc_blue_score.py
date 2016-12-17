from bleu import calculate_bleu_score
import sys
import json
ref_path=sys.argv[1].strip()
sys_path=sys.argv[2].strip()

d_ref = []
d_pred = []

with open(ref_path, 'r') as f:
    d_ref = json.load(f)

with open(sys_path, 'r') as f:
    d_pred = json.load(f)

sum_bleu = 0
for i, ref in enumerate(d_ref):
    ref_caption = ref['caption']
    pred_caption = d_pred[i]['caption']
    print('Comparing %s and %s'%(ref_caption, pred_caption))
    b_score = calculate_bleu_score(ref_caption, pred_caption)
    print(b_score)
    sum_bleu += b_score

avg_bleu = sum_bleu/len(d_ref)
print("Avg Bleu -  %s" %avg_bleu)

#print(calculate_bleu_score(ref_path, sys_path))
