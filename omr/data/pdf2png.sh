#!/bin/bash

POOL=8
ctr=0
for f in `ls out1`; do
    echo $f
    f_basename=$(basename "$f" .pdf)
    ctr=$((ctr+1))
    convert -density 200 -quality 100 -transparent white out1/$f out1png/$f_basename.png &
    if [ $ctr -eq 4 ]; then
        wait
        ctr=0
    fi
done
