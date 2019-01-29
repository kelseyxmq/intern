import pandas as pd

# replace the filenames!
df1 = pd.read_csv('file1.csv')
df2 = pd.read_csv('file2.csv')
df3 = pd.read_csv('file3.csv')

def vote(l):
    yes = 0
    no = 0
    for item in l:
        if item == 'yes':
            yes += 1
        else:
            no += 1
    if yes > no:
        return 'yes'
    else:
        return 'no'

ret = df1
for i in range(len(df1)):
    l = []
    l.append(df1.loc[i, 'is_same'])
    l.sppend(df2.loc[i, 'is_same'])
    l.append(df3.loc[i, 'is_same'])
    ret.loc[i, 'is_same'] = vote(l)

# if extra column, uncomment this line
# ret.drop('Unnamed 0', axis = 1)
ret.to_csv('result.csv', index = False)