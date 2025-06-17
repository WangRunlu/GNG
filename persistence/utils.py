import pickle
import numpy as np
import networkx as nx
from gudhi import SimplexTree
from persistence.bfs_looping import BFSLoopDetection, get_directed_intersection_attribute, get_polygon_loop
from bresenham import bresenham
import math

def print_complex_attributes(cmplx):
    """Pretty‑print a few basic statistics of a GUDHI SimplexTree.

    Works with both old and new GUDHI versions (where ``get_skeleton`` now
    returns a generator).
    """
    print(f"num_vertices={cmplx.num_vertices()}")
    print(f"num_simplices={cmplx.num_simplices()}")

    # get_skeleton(k) → generator in newer GUDHI; cast to list so len() works
    skel2 = list(cmplx.get_skeleton(2))
    if len(skel2) < 20:
        print("skeleton(2) =")
        for sk_value in skel2:
            print(sk_value)

import pickle
import numpy as np

def load_hilbert_map(map_type='intel'):
    """
    :param map_type: type of map to load, eg "intel", "freiburg", etc.
    :return: a dictionary with key 'Xq' as locations and key 'yq' as occupancy probabilities
    """
    mapdata = {}
    resolution = None
    if map_type == "drive":
        resolution = 0.5
        with open('./../dataset/mapdata_0.pickle', 'rb') as tf:
            mapdata_ = pickle.load(tf)
            mapdata['Xq'] = np.array(mapdata_['Xq'])
            mapdata['yq'] = np.array(mapdata_['yq'])
        for i in range(1, 4):
            with open('./../dataset/mapdata_{}.pickle'.format(i), 'rb') as tf:
                mapdata_ = pickle.load(tf)
                mapdata['Xq'] = np.concatenate((mapdata['Xq'], np.array(mapdata_['Xq'])), axis=0)
                mapdata['yq'] = np.concatenate((mapdata['yq'], np.array(mapdata_['yq'])), axis=0)
    elif map_type == "freiburg":
        resolution = 0.2
        with open('./../dataset/mapdata_4798.pickle', 'rb') as tf:
            mapdata_ = pickle.load(tf)
            mapdata['Xq'] = np.array(mapdata_['X'])
            mapdata['yq'] = np.array(mapdata_['Y'])
    elif map_type == "fhw":
        resolution = 0.2
        with open('./../dataset/mapdata_499.pickle', 'rb') as tf:
            mapdata_ = pickle.load(tf)
            mapdata['Xq'] = np.array(mapdata_['X'])
            mapdata['yq'] = np.array(mapdata_['Y'])
    elif map_type == "bhm":
        resolution = 0.3
        with open('./../dataset/bhm_model.pickle', 'rb') as tf:
            mapdata_ = pickle.load(tf)
            if 'Xq' in mapdata_ and 'yq' in mapdata_:
                mapdata['Xq'] = np.array(mapdata_['Xq'])
                mapdata['yq'] = np.array(mapdata_['yq'])
            else:
                import numpy as np, math
                from sklearn.metrics.pairwise import rbf_kernel
                if 'grid' in mapdata_ and 'mu' in mapdata_ and 'sig' in mapdata_:
                    grid = np.array(mapdata_['grid'])
                    mu = np.array(mapdata_['mu'])
                    sig = np.array(mapdata_['sig'])
                    gamma = mapdata_.get('gamma', 1.0)
                    Phi = rbf_kernel(grid, grid, gamma=gamma)
                    mu_a = Phi.dot(mu[1:]) + mu[0]
                    sig2_inv_a = np.sum((Phi ** 2) * sig[1:], axis=1)
                    kappa = 1.0 / np.sqrt(1 + np.pi * sig2_inv_a / 8)
                    occ_prob = 1.0 / (1 + np.exp(-kappa * mu_a))
                    mapdata['Xq'] = grid
                    mapdata['yq'] = occ_prob
                else:
                    raise KeyError("bhm_model.pickle must contain either Xq/yq or grid+params")
    else:  # default: intel
        resolution = 0.3
        with open('./../dataset/intel_mapdata.pickle', 'rb') as tf:
            mapdata_ = pickle.load(tf)
            if 'Xq' in mapdata_ and 'yq' in mapdata_:
                mapdata['Xq'] = np.array(mapdata_['Xq'])
                mapdata['yq'] = np.array(mapdata_['yq'])
            elif 'X' in mapdata_ and 'Y' in mapdata_:
                mapdata['Xq'] = np.array(mapdata_['X'])
                mapdata['yq'] = np.array(mapdata_['Y'])
            else:
                raise KeyError("intel_mapdata.pickle must contain 'Xq','yq' or 'X','Y'.")
    return mapdata, resolution


# 新增: 将map_dict转换为二维数组表示
def convert_map_dict_to_array(map_dict, resolution):
    """
    :param map_dict: a dictionary with key 'Xq' as locations and key 'yq' as occupancy probabilities
    :param resolution: 每个格子的分辨率
    :return: hilbert map in array form
    Note: 默认 hard code 输出 600x600 区域，可以根据实际需要调整
    """
    map_array = np.zeros([600, 600]) + 0.5
    for indx, point in enumerate(map_dict['Xq']):
        x_idx = int(point[0] * (1 / resolution) + 300)
        y_idx = int(point[1] * (1 / resolution) + 300)
        if 0 <= x_idx < 600 and 0 <= y_idx < 600:
            map_array[x_idx][y_idx] = map_dict['yq'][indx]
    return map_array
def collision_check(map_array, pos1, pos2, obstacle_threshold, resolution):
    """
    检查两点连线之间是否有障碍物
    """
    from bresenham import bresenham
    ipos1 = [int(pos1[0] * (1 / resolution) + 300), int(pos1[1] * (1 / resolution) + 300)]
    ipos2 = [int(pos2[0] * (1 / resolution) + 300), int(pos2[1] * (1 / resolution) + 300)]
    check_list = list(bresenham(ipos1[0], ipos1[1], ipos2[0], ipos2[1]))
    for cell in check_list:
        if map_array[cell[0]][cell[1]] > obstacle_threshold and (not np.isnan(map_array[cell[0]][cell[1]])):
            return False
    return True

def convert_gng_to_nxgng(g, map_array, obs_threshold, resolution):
    """
    把GNG内部结构转成networkx的Graph，方便后续拓扑分析
    """
    nxgraph = nx.Graph()
    nodeid = {}
    for indx, node in enumerate(g.graph.nodes):
        nodeid[node] = indx
        nxgraph.add_node(nodeid[node], pos=(node.weight[0][0], node.weight[0][1]))
    positions = nx.get_node_attributes(nxgraph, "pos")
    for node_1, node_2 in g.graph.edges:
        if collision_check(map_array, positions[nodeid[node_1]], positions[nodeid[node_2]], obs_threshold, resolution):
            nxgraph.add_edge(nodeid[node_1], nodeid[node_2])
    return nxgraph
def get_topological_accuracy(gng_, feature, local_distance):
    """
    判断每个特征点feature是不是被GNG结构正确环绕（拓扑准确性检测，1-hom）
    """
    topological_accuracy = []
    position = nx.get_node_attributes(gng_, 'pos')
    for f_indx, f in enumerate(feature):
        # 找最近节点
        short_distance = 10000
        closest_node = None
        local_graph = nx.Graph()
        for indx, node in enumerate(gng_.nodes):
            pose = position[node]
            distance = math.sqrt((f[0] - pose[0]) ** 2 + (f[1] - pose[1]) ** 2)
            if short_distance > distance:
                closest_node = indx
                short_distance = distance
            if local_distance > distance:
                local_graph.add_node(indx, pos=pose)

        for node1, node2 in gng_.edges:
            if (node1 in local_graph.nodes) and (node2 in local_graph.nodes):
                local_graph.add_edge(node1, node2)

        loopfinder = BFSLoopDetection()
        loops = loopfinder.get_breadth_first_search_loops(local_graph, closest_node)
        collisions = get_directed_intersection_attribute(local_graph, f)
        is_inside, polygon = get_polygon_loop(loops, collisions)
        topological_accuracy.append(is_inside)
    return topological_accuracy

def count_components(g):
    is_connected = False
    explored = {}
    for node in g.nodes:
        explored[node] = False
    num_connected_components = 0
    while not all(value == True for value in explored.values()):
        for key, value in explored.items():
            if not value:
                start = key
                break
        queue = [start]
        explored[start] = True
        node_explored_count = 0
        while len(queue) != 0:
            node = queue.pop(0)
            for adj in g.neighbors(node):
                if not explored[adj]:
                    explored[adj] = True
                    queue.append(adj)
            node_explored_count += 1
        if node_explored_count > 1:
            num_connected_components += 1
    if num_connected_components == 1:
        is_connected = True
    return is_connected, num_connected_components

def get_0hom_topological_accuracy(gng_, feature, local_distance):
    """
    判断每个特征点feature在local_distance范围内的GNG局部子图是否是连通的（0-hom）
    """
    topological_accuracy_0hom = []
    position = nx.get_node_attributes(gng_, 'pos')
    for f_indx, f in enumerate(feature):
        local_graph = nx.Graph()
        for indx, node in enumerate(gng_.nodes):
            pose = position[node]
            distance = math.sqrt((f[0] - pose[0]) ** 2 + (f[1] - pose[1]) ** 2)
            if local_distance > distance:
                local_graph.add_node(indx, pos=pose)

        for node1, node2 in gng_.edges:
            if (node1 in local_graph.nodes) and (node2 in local_graph.nodes):
                local_graph.add_edge(node1, node2)

        is_connected, num_components = count_components(local_graph)
        topological_accuracy_0hom.append(is_connected)
    return topological_accuracy_0hom
