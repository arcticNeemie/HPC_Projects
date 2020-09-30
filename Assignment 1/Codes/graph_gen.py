import matplotlib.pyplot as plt
import sys

file = sys.argv[1]
title = sys.argv[2]
n = range(int(sys.argv[1]))
title = sys.argv[int(sys.argv[1])+2]

first = False
for i in n:
    if first:
        plt.plotfile(sys.argv[i+2], ('n', 'time'), newfig= False, label = sys.argv[i+2][:-4])
    else:
        first = True
        plt.plotfile(sys.argv[i+2], ('n', 'time'), label = sys.argv[i+2][:-4])



# plt.plotfile(file, ('input', 'time'))
plt.xlabel('Query Points (n)')
plt.ylabel('Time (s)')
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.legend(shadow = False, fancybox = True)
plt.savefig(title +'.png')
