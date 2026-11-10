#!/bin/bash
# Usage: tools/compare_regimes.sh <run-name>    e.g. t113-fix5-f5-c3
# Evaluates every checkpoint of a3c/<run> + 3 reference checkpoints under EVAL and TRAIN
# conditions with bc_extract (1 sampled episode, seed 1000). Legacy single-seed check;
# the multi-seed comparison is tools/eval_harness.py.
# Needs the load servers up: docker compose up -d loadserver_0 loadserver_1 loadserver_2 loadserver_3
cd "$(dirname "$0")/.." || exit 1          # repo root
N=${1:?give a run name, e.g. t113-fix5-f5-c3}
R=results/raw/eval                          # bc_data/ and bc_logs/ live here
mkdir -p $R/bc_data $R/bc_logs

echo "== training run $N"; grep "done in" logs/$N.log
[ -f a3c/$N/train_conditions.json ] && tr -d '\n ' < a3c/$N/train_conditions.json && echo
echo "== mean training reward/step per 100 episodes"
grep " Ep: " logs/$N.log | awk '{s+=$NF; n++} n==100 {printf "eps %4d-%4d  %.3f\n", $3-99, $3, s/n; s=0; n=0}'

run() {  # model ckpt slo budget_cpu budget_mem log_dir out_jsonl
  docker compose run --rm -T tools python3 tools/bc_extract.py --teacher a3c/$1/model/model$2.pt \
    --out $7 --nconf 5 --ncomp 3 --seed 1000 --slo $3 --budget $4 $5 --episodes 1 \
    --log_dir $6 < /dev/null >/dev/null 2>&1
  tail -1 $6/f5-c3.csv | tr -d '\r' | awk -F, '{printf "%5s %8.1f %4s %4s", $2, $4, $7, $8}'
}

CKPTS=$(ls a3c/$N/model | sed 's/model\([0-9]*\)\.pt/\1/' | sort -n)
echo
printf "%-24s | %-25s | %-25s\n" "" " EVAL  slo150 [2500,3600]" " TRAIN slo250 [600,800]"
printf "%-24s | %5s %8s %4s %4s | %5s %8s %4s %4s\n" model steps reward brch ovr steps reward brch ovr
for M in $(for k in $CKPTS; do echo "$N:$k"; done) fix23-f5-c3:100 fix23-f5-c3:200 repro-f5-c3:100; do
  n=${M%:*}; k=${M#*:}
  rm -f $R/bc_data/${n}_m$k.jsonl; rm -rf $R/bc_logs/$n/m$k $R/bc_logs/trainreg/${n}_m$k
  e=$(run $n $k 150 2500 3600 $R/bc_logs/$n/m$k $R/bc_data/${n}_m$k.jsonl)
  t=$(run $n $k 250 600 800 $R/bc_logs/trainreg/${n}_m$k /tmp/tr.jsonl)
  printf "%-24s | %s | %s\n" "$n m$k" "$e" "$t"
done

echo; echo "== $N policy shape (eval conditions)"
docker compose run --rm -T tools python3 tools/bc_series.py $N f5-c3 < /dev/null 2>/dev/null
