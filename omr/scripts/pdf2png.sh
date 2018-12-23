#!/bin/bash

POOL=8
ctr=0
for f in `ls smi2pdf`; do
    echo $f
    f_basename=$(basename "$f" .pdf)
    ctr=$((ctr+1))
    convert -density 200 -quality 100 -transparent white smi2pdf/$f smi2png/$f_basename.png &
    if [ $ctr -eq 4 ]; then
        wait
        ctr=0
    fi
done
