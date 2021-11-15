import os

for subdir, dirs, files in os.walk('./aligned/'):
    for filename in files:
        tmp = filename.split('.')
        if len(tmp)>2:
            filename_new = tmp[-2] + '.' + tmp[-1]
            os.rename(os.path.join(subdir,filename), os.path.join(subdir,filename_new))
