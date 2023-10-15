from ripser import ripser
import numpy as np
import math
from itertools import combinations
from scipy.spatial.distance import cdist
from gudhi.rips_complex import RipsComplex
from pprint import pprint
import time
import pandas as pd
from sklearn.linear_model import LinearRegression

class RipserProcesser:
    def __init__(self, X, max_dim, distance_matrix=False):
        self.N = X.shape[0]
        if distance_matrix:
            self.dist_mat = X
        else:
            self.dist_mat = cdist(X, X)

        # 二項係数を前計算しておく
        self.binomial_table = [[math.comb(m, r) for r in range(m+1)] for m in range(self.N+1)] 
        # ripser の実行
        output = ripser(self.dist_mat, max_dim, distance_matrix=True, do_cocycles=True)
        self.dgms = output["dgms"]

        # cocycle を処理し，掃き出しに用いられた行列を取得
        self.cocycle_list = [[] for _ in range(max_dim+1)]
        for dim in range(1, max_dim+1):
            for col in output["cocycles"][dim]:
                tmp = set()
                # print(col)
                for simp in col:
                    tmp.add(self.get_simplex_index(list(simp[:-1])))
                self.cocycle_list[dim].append(tmp)
        
        # coboundary に掃き出し行列をかけて，birth death ペアを取得
        self.persistence_pairs = [[] for _ in range(max_dim+1)]
        for dim in range(1, max_dim+1):
            for col in self.cocycle_list[dim]:
                tmp = set()
                for simp_idx in col:
                    self.add_coboundary(dim, simp_idx, tmp)
                death_simp_idx = - min(tmp)[1]
                birth_simp_idx = min(col, key=lambda x: (self.get_appear_time(dim, x), - x))
                self.persistence_pairs[dim].append((birth_simp_idx, death_simp_idx))
    
    def binomial(self, m, r):
        if m < r or r < 0:
            return 0
        return self.binomial_table[m][r]
    
    def get_simplex_index(self, vertice_list):
        vertice_list.sort(reverse=True)
        dim = len(vertice_list) - 1
        ret = sum([self.binomial(vertice_list[i], dim-i+1) for i in range(dim+1)])
        # assert self.get_simplex_vertices(dim, ret) == vertice_list
        return ret
    
    def get_max_vertex(self, simplex_dim, simplex_idx, right):
        left = simplex_dim
        while right - left > 1:
            mid = (left + right) // 2
            if self.binomial(mid, simplex_dim+1) <= simplex_idx:
                left = mid
            else:
                right = mid
        return left
    
    def get_simplex_vertices(self, simplex_dim, simplex_idx):
        ret = []
        dim, idx = simplex_dim, simplex_idx
        while len(ret) <= simplex_dim:
            ret.append(self.get_max_vertex(dim, idx, ret[-1] if ret else self.N))
            idx -= self.binomial(ret[-1], dim+1)
            dim -= 1
        return ret
        
    def get_appear_time(self, simplex_dim, simplex_idx):
        vertex_list = self.get_simplex_vertices(simplex_dim, simplex_idx)
        ret = 0
        for u, v in combinations(vertex_list, 2):
            ret = max(ret, self.dist_mat[u, v])
        return ret
    
    # ある単体のcoboundaryとなる単体番号を取得し，指定したsetにaddする
    def add_coboundary(self, simplex_dim, simplex_idx, target_set):
        simplex = self.get_simplex_vertices(simplex_dim, simplex_idx)
        k = simplex_dim
        term_1 = 0
        term_2 = sum([self.binomial(simplex[-l-1], l+1) for l in range(simplex_dim+1)])
        for j in range(self.N-1, -1, -1):
            if j == simplex[-k-1]:
                # term_2 から現在の k に対応する項を引く
                if k >= 0:
                    term_2 -= self.binomial(simplex[-k-1], k+1)
                else:
                    term_2 = 0 
                # kが更新される
                k -= 1
                # term_1 に新しい k に対応する項を足す
                term_1 += self.binomial(simplex[-k-2], k+3)
                continue
            coface_idx = term_1 + self.binomial(j, k+2) + term_2
            target_set ^= {(self.get_appear_time(simplex_dim+1, coface_idx), - coface_idx)}

if __name__ == "__main__":
    # 結果が正しいことを確認 & 実行時間を計測
    num_points_list = [20, 40, 60, 80, 100, 150, 200, 250, 300]
    time_info = {"ripser_1-PH": [0] * len(num_points_list),
                 "ripser_2-PH": [0] * len(num_points_list),
                 "gudhi_1-PH": [0] * len(num_points_list),
                 "gudhi_2-PH": [0] * len(num_points_list),
                 }
    for i, num_points in enumerate(num_points_list):
        for trial in range(5):
            X = np.random.random((num_points, 3))

            ## ripser_1-PH ##
            start = time.time()
            ripser_processer = RipserProcesser(X, 1)
            ripser_barcode = []
            for birth, death in ripser_processer.persistence_pairs[1]:
                ripser_barcode.append((ripser_processer.get_appear_time(1, birth), ripser_processer.get_appear_time(2, death)))
            time_info["ripser_1-PH"][i] += time.time() - start
            # 正しいことの確認
            ripser_barcode.sort()
            genuine_ripser_output = ripser(X, maxdim=1)
            genuine_ripser_barcode = sorted(list(map(tuple, genuine_ripser_output["dgms"][1])))
            if all([(x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 < 1e-3 for x, y in zip(ripser_barcode, genuine_ripser_barcode)]):
                print(f"num_points={num_points}, trial={trial}, ripser_1-PH: OK", flush=True)
            else:
                print(f"num_points={num_points}, trial={trial}, ripser_1-PH: NG", flush=True)
            
            ## gudhi_1-PH ##
            start = time.time()
            rips = RipsComplex(points=X)
            simplex_tree = rips.create_simplex_tree(max_dimension=2)
            gudhi_barcode = simplex_tree.persistence()
            simplex_tree.persistence_pairs()
            time_info["gudhi_1-PH"][i] += time.time() - start
            # ripser と gudhi の結果が一致していることを確認
            gudhi_barcode = sorted([x[1] for x in gudhi_barcode if x[0] == 1])
            if all([(x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 < 1e-3 for x, y in zip(genuine_ripser_barcode, gudhi_barcode)]):
                print(f"num_points={num_points}, trial={trial}, gudhi_1-PH: OK", flush=True)
            else:
                print(f"num_points={num_points}, trial={trial}, gudhi_1-PH: NG", flush=True)

            ## ripser_2-PH ##
            start = time.time()
            ripser_processer = RipserProcesser(X, 2)
            ripser_barcode = []
            for birth, death in ripser_processer.persistence_pairs[2]:
                ripser_barcode.append((ripser_processer.get_appear_time(2, birth), ripser_processer.get_appear_time(3, death)))
            time_info["ripser_2-PH"][i] += time.time() - start
            # 正しいことの確認
            ripser_barcode.sort()
            genuine_ripser_output = ripser(X, maxdim=2)
            genuine_ripser_barcode = sorted(list(map(tuple, genuine_ripser_output["dgms"][2])))
            if all([(x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 < 1e-3 for x, y in zip(ripser_barcode, genuine_ripser_barcode)]):
                print(f"num_points={num_points}, trial={trial}, ripser_2-PH: OK", flush=True)
            else:
                print(f"num_points={num_points}, trial={trial}, ripser_2-PH: NG", flush=True)

            ## gudhi_2-PH ##
            start = time.time()
            rips = RipsComplex(points=X)
            simplex_tree = rips.create_simplex_tree(max_dimension=3)
            simplex_tree.compute_persistence()
            simplex_tree.persistence_pairs()
            time_info["gudhi_2-PH"][i] += time.time() - start
            # ripser と gudhi の結果が一致していることを確認
            gudhi_barcode = sorted([x[1] for x in gudhi_barcode if x[0] == 2])
            if all([(x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 < 1e-3 for x, y in zip(genuine_ripser_barcode, gudhi_barcode)]):
                print(f"num_points={num_points}, trial={trial}, gudhi_2-PH: OK", flush=True)
            else:
                print(f"num_points={num_points}, trial={trial}, gudhi_2-PH: NG", flush=True)
        
        time_info["ripser_1-PH"][i] /= 5

    pprint(time_info)

    # 見やすいように pd.DataFrame に変換後，markdown形式で出力
    df = pd.DataFrame(time_info, index=num_points_list)
    df.index.name = "num_points"
    df.to_markdown("result/ripser_processor/time_info.md")

    # log 点の数 - log 実行時間 で線形回帰
    X = np.log(np.array(num_points_list)).reshape(-1, 1)
    for key in time_info.keys():
        y = np.log(np.array(time_info[key]))
        model = LinearRegression()
        model.fit(X, y)
        print(f"{key}: {float(np.exp(model.intercept_))} * n ^ {float(model.coef_[0])}")