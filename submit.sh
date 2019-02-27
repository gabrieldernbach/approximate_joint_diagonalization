#!/usr/bin/env bash
#$ -l cuda=2
#$ -q all.q
#$ -cwd
#$ -V
#$ -t 1

source testenv/bin/activate
python distributed_pool.py
echo 'hi'
