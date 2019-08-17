module load cuda55/toolkit/5.5.22
module load prun

prun -np 1 -native '-l gpu=GTX480' myconvolution 0
prun -np 1 -native '-l gpu=GTX480' myconvolution 1
