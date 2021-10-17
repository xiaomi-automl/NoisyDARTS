#!/bin/sh

for v in {a..n};do
    python vis_cell.py noisy_darts_$v
done
