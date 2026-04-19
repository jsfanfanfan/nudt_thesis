# -*- coding: utf-8 -*-
filepath = r'c:\Users\fanji\Desktop\毕业论文\nudt_thesis\data\chap03.tex'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()
if '基于 MD 的筛选机制在匹配质量与求解效率之间取得折中' in content:
    print('NEW content confirmed')
elif '在实现层面，规划器对所有设备拓扑计算 MD' in content:
    print('OLD content still present - FAILED')
