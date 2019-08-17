#!/bin/bash

module load prun
module load cuda55/toolkit/5.5.22

prun -v -1 -np 1 -native '-l gpu=GTX480' myhistogram 2 2097152 1024

prun -v -1 -np 1 -native '-l gpu=GTX480' myhistogram 2 4194304 1024
prun -v -1 -np 1 -native '-l gpu=GTX480' myhistogram 2 8388608 1024
prun -v -1 -np 1 -native '-l gpu=GTX480' myhistogram 2 16777216 1024
prun -v -1 -np 1 -native '-l gpu=GTX480' myhistogram 2 33554432 1024



