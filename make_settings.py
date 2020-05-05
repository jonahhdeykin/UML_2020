import random
import math
import sys


eta_s = random.random()
gam_s = random.random()
gem_m_s = random.random()
gem_s_s = random.random()


eta_range = [0.00001, 100]
gam_range = [0.1, 10]
gem_m_range = [0.01, 0.99]
gem_s_range = [1, 10000]


eta_val = math.exp(math.log(eta_range[0]) + (math.log(eta_range[1]) - math.log(eta_range[0])) * eta_s )
gam_val = math.exp(math.log(gam_range[0]) + (math.log(gam_range[1]) - math.log(gam_range[0])) * gam_s )
gem_m_val = math.exp(math.log(gem_m_range[0]) + (math.log(gem_m_range[1]) - math.log(gem_m_range[0])) * gem_m_s)
gem_s_val = math.exp(math.log(gem_s_range[0]) + (math.log(gem_s_range[1]) - math.log(gem_s_range[0])) * gem_s_s)


with open(sys.argv[1], 'w') as f:

	f.write('DEPTH 8\n')
	f.write('ETA')
	for x in range(0, 8):
		f.write(' ' + str(eta_val))
	f.write('\n')
	f.write('GAM')
	for x in range(0, 7):
		f.write(' ' + str(gam_val))
	f.write('\n')
	f.write('GEM_MEAN {}\n'.format(gem_m_val))
	f.write('GEM_SCALE {}\n'.format(gem_s_val))
	f.write('SCALING_SHAPE 1.0\n')
	f.write('SCALING_SCALE 0.5\n')
	f.write('SAMPLE_ETA 1\n')
	f.write('SAMPLE_GEM 1')
