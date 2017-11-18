import os
import os.path
import argparse
def generatesyssetwords(datadir):
    subdir=os.listdir(datadir)
    sysnsetfile=open(datadir+'/'+'synset.txt','w')
    for sub in subdir:
        if os.path.isdir(datadir+'/'+sub):
            print sub
            sysnsetfile.writelines(sub+'\n')

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='generate synsetwords')
    parser.add_argument('--datadir', type=str, default='chars',help='path to folder that contain datasets.')
    args = parser.parse_args()
    generatesyssetwords(args.datadir)
if __name__ == '__main__':
    main()