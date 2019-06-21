import sys

with open('./../data/sample_submission.csv') as f:
    line = f.readlines()

with open(sys.argv[2],'w') as f:
    f.write(line[0])
    with open(sys.argv[1]) as f_in:
        for i,pre in enumerate(f_in):
            line[i+1] = line[i+1].split(',')
            f.write("{0},{1}".format(line[i+1][0],pre))

