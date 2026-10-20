#!/bin/bash
# Evaluate every epoch checkpoint of 30ogb-cube_lewm, spread over 4 GPUs.
source ./env.sh
NGPU=4
run() {
  grep -q "success_rate" eval_logs/epoch_$1.log 2>/dev/null && return
  e=$1; gpu=$(( (e-1) % NGPU ))
  CUDA_VISIBLE_DEVICES=$gpu python eval.py --config-name=cube.yaml \
    policy=30ogb-cube_lewm/weights_epoch_$e.pt output.filename=results_epoch_$e.txt \
    > eval_logs/epoch_$e.log 2>&1
}
for g in 0 1 2 3; do
  ( for e in $(seq $((g+1)) $NGPU 30); do run $e; done ) &
done
wait
python - <<'PY'
import re,glob
rows=[]
for e in range(1,31):
    try: t=open(f"eval_logs/epoch_{e}.log").read()
    except FileNotFoundError: t=""
    m=re.findall(r"success_rate'?\"?:\s*(?:np\.float64\()?([\d.]+)",t)
    rows.append((e,m[-1] if m else "FAILED"))
with open("epoch_success_rates.txt","w") as f:
    f.write("epoch\tsuccess_rate\n")
    for e,s in rows: f.write(f"{e}\t{s}\n")
print(open("epoch_success_rates.txt").read())
PY
