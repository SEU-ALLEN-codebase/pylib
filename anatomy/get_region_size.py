##########################################################
#Author:          Yufeng Liu
#Create time:     2026-01-13
#Description:               
##########################################################
import numpy as np
import json
from collections import defaultdict

# 假设你的库和函数已正确导入
from anatomy.anatomy_config import MASK_CCF25_FILE
from anatomy.anatomy_core import parse_ana_tree
from file_io import load_image

# 1. 加载数据
print("Loading mask image and anatomy tree...")
ccf25_mask = load_image(MASK_CCF25_FILE)  # 3D mask, 值应为脑区ID
ana_tree = parse_ana_tree()  # 返回字典: {脑区ID: 脑区信息字典}

print(f"Mask shape: {ccf25_mask.shape}")
print(f"Anatomy tree entries: {len(ana_tree)}")

# 2. 计算每个脑区ID在mask中直接出现的体素数
print("Calculating direct voxel counts for each region ID...")
id_to_direct_voxels = {}
unique_ids, counts = np.unique(ccf25_mask, return_counts=True)
for uid, cnt in zip(unique_ids, counts):
    # 跳过背景（ID为0）
    if uid != 0:
        id_to_direct_voxels[int(uid)] = int(cnt)

# 3. 构建脑区ID到其所有后代ID的映射（基于 structure_id_path）
print("Building region hierarchy (parent -> all descendants)...")
# 第一步：构建直接父子关系（parent -> direct children）
parent_to_children = defaultdict(list)
# 第二步：构建祖先到所有后代的完整关系
id_to_all_descendants = defaultdict(list)

for region_id, region_info in ana_tree.items():
    path = region_info['structure_id_path']
    # 当前脑区的直接父节点是路径中倒数第二个ID（如果存在）
    if len(path) > 1:
        parent_id = path[-2]
        parent_to_children[parent_id].append(region_id)
    # 对于路径中的每一个祖先，当前region_id都是其后代
    for ancestor_id in path[:-1]:  # 不包括自己
        id_to_all_descendants[ancestor_id].append(region_id)

# （可选）验证：对于某些脑区，打印其后代数量
# for rid, desc in list(id_to_all_descendants.items())[:5]:
#     print(f"Region {rid} has {len(desc)} descendant(s)")

# 4. 为每个脑区计算总体积 = 直接体素 + 所有后代的直接体素
print("Computing total voxel count for each region...")
results = []

for region_id, region_info in ana_tree.items():
    acronym = region_info['acronym']
    name = region_info['name']
    
    # 4.1 该脑区自身的直接体素数（可能在mask中不存在）
    direct_voxels = id_to_direct_voxels.get(region_id, 0)
    
    # 4.2 获取该脑区的所有后代ID列表
    descendant_ids = id_to_all_descendants.get(region_id, [])
    
    # 4.3 计算所有后代的直接体素之和
    descendant_voxels = 0
    for desc_id in descendant_ids:
        descendant_voxels += id_to_direct_voxels.get(desc_id, 0)
    
    # 4.4 总体积 = 自身直接体素 + 后代直接体素
    total_voxels = direct_voxels + descendant_voxels
    
    # 保存结果
    results.append({
        'id': region_id,
        'acronym': acronym,
        'name': name,
        'direct_voxels': direct_voxels,
        'descendant_voxels': descendant_voxels,
        'total_voxels': total_voxels
    })

# 5. 按ID排序并输出
results.sort(key=lambda x: x['id'])

# 6. 保存为CSV文件（易于查看和处理）
output_file = "brain_region_volumes.csv"
print(f"Saving results to {output_file}...")

with open(output_file, 'w') as f:
    # 写入表头
    f.write("id,acronym,name,direct_voxels,descendant_voxels,total_voxels\n")
    for r in results:
        f.write(f"{r['id']},{r['acronym']},{r['name']},{r['direct_voxels']},{r['descendant_voxels']},{r['total_voxels']}\n")

print("Done!")
print(f"Total regions processed: {len(results)}")
print(f"Regions with non-zero total voxels: {sum(1 for r in results if r['total_voxels'] > 0)}")

# 7. 可选：打印前20个脑区的结果作为验证
print("\n--- First 20 regions (sorted by ID) ---")
for r in results[:20]:
    print(f"ID {r['id']:4d} | {r['acronym']:10s} | {r['name']:35s} | Direct: {r['direct_voxels']:8d} | Descendant: {r['descendant_voxels']:8d} | Total: {r['total_voxels']:8d}")
