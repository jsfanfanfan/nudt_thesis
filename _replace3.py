# -*- coding: utf-8 -*-
filepath = r'c:\Users\fanji\Desktop\毕业论文\nudt_thesis\data\chap03.tex'

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Verify target lines exist
assert '在实现层面，规划器对所有设备拓扑计算 MD' in lines[183], f"Line 184 mismatch: {lines[183][:30]}"

new_lines = [
    '基于 MD 的筛选机制在匹配质量与求解效率之间取得折中：'
    '一方面，仅将 MD 较高的前一半拓扑传入后续整数线性规划阶段，'
    '避免在供需趋势严重不一致的拓扑上浪费求解时间；'
    '另一方面，保留多个候选而非仅取最优，'
    '为冻结感知划分步骤提供了充分的搜索空间，'
    '使其能够在不同冻结场景下分别找到更合适的划分方案。\n',
    '从整体设计目标来看，需求驱动的设备拓扑识别使设备排列的选取不再依赖显存或算力的单一维度排序，'
    '而是直接由 MLLMs 的计算—显存需求变化趋势驱动，'
    '从而使后续划分能够更好地适应多模态大语言模型的模块异构性与分阶段冻结训练带来的动态需求变化。\n',
]

lines[183:186] = new_lines

with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('SUCCESS')
