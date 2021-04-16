import os
from os import listdir
# 바운딩 박스 잡힌것만 파악해오기
def read_dataset(dir):
    images = []
    annotations = []
    bounding_box = []
    delete_list = []
    hwc = []

    for file in listdir(dir):
        if 'jpg' in file.lower():
            if os.path.getsize(dir + file.replace(file.split('.')[-1], 'txt')) == 0:
                delete_list.extend([file, file.replace(file.split('.')[-1], 'txt')])

    # 만약 빈 텍스트 파일이 있으면 (바운딩박스가 잘 안잡혔으면) 삭제
    if delete_list:
        for i in delete_list:
            os.remove(dir+i)