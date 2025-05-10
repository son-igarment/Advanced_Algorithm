"""
Dynamic Distributed Ant Colony System (DDACS) Algorithm
for Resource-Constrained Project Scheduling Problems

Author: Phạm Lê Ngọc Sơn
"""

import numpy as np
import random

# Hàm tính E_j và L_j
def calculate_E_L(predecessors, successors, p, N):
    E = [0] * (N + 2)
    L = [float('inf')] * (N + 2)
    L[N + 1] = 0  # Hoạt động kết thúc

    # Tính E_j
    for j in range(1, N + 1):
        if not predecessors[j]:
            E[j] = 0
        else:
            E[j] = max(E[i] + p[i] for i in predecessors[j])

    # Tính L_j
    for j in range(N, -1, -1):
        if not successors[j]:
            L[j] = E[j] + p[j]
        else:
            L[j] = min(L[i] - p[j] for i in successors[j])

    return E, L

def nth_root(x, n):
    return round(x ** (1 / n))

# Lớp DDACS
class DDACS:
    def __init__(self, N, T, c, c1, ant, alpha, beta, rho, delta, q0, q1, max_iter, p, R, r, predecessors, successors):
        self.N = N  # Số hoạt động
        self.T = T  # Thời gian tối đa
        self.c = c
        self.c1 = c1
        self.ant = ant  # Số kiến
        self.alpha = alpha
        self.beta = beta
        self.rho = rho  # Tỷ lệ bay hơi cục bộ
        self.delta = delta  # Tỷ lệ cập nhật toàn cục
        self.q0 = q0
        self.q1 = q1
        self.max_iter = max_iter
        self.p = p  # Thời gian thực hiện
        self.R = R  # Giới hạn tài nguyên
        self.r = r  # Yêu cầu tài nguyên
        self.predecessors = predecessors
        self.successors = successors
        self.tau = np.ones((T, N + 2)) * 0.1  # Ma trận pheromone
        self.E, self.L = calculate_E_L(predecessors, successors, p, N)
        self.tau0 = 0.01  # Giá trị pheromone ban đầu

    # Hàm heuristic eta
    def eta(self, t, p_j, E_j, L_j):
        d_j = abs(L_j - t)
        if E_j <= t < L_j:
            denominator = (d_j + 1) * nth_root(p_j, self.c)
        elif t >= L_j:
            denominator = (2 - d_j / self.c1) * nth_root(p_j, self.c)
        else:  # current_time < Ej
            return 1e-6

        return 1 / denominator if denominator > 0 else 1e-6

    def check_resource(self, t, j, solution):
        """Kiểm tra tài nguyên có đủ cho hoạt động j tại t"""
        for tt in range(t, t + self.p[j]):
            if tt >= self.T:
                return False
            used = np.zeros(len(self.R))
            for i in range(self.N + 2):
                if solution[i] != -1 and solution[i] <= tt < solution[i] + self.p[i]:
                    used += self.r[i]
            if any(used[k] + self.r[j][k] > self.R[k] for k in range(len(self.R))):
                return False
        return True

    def build_solution(self):
        """Xây dựng giải pháp cho một kiến"""
        solution = [-1] * (self.N + 2)  # Thời gian bắt đầu của mỗi hoạt động
        solution[0] = 0  # Hoạt động bắt đầu
        C = {0}  # Tập hợp hoạt động đã lên lịch
        t = 0

        while len(C) < self.N + 2:
            if t >= self.T:  # Kiểm tra nếu t vượt quá kích thước của self.tau
                break  # Thoát vòng lặp nếu t không hợp lệ

            # Tập hợp các hoạt động có thể thực hiện J_k, bao gồm các hoạt động:
            # - Đã hoàn thành tất cả các hoạt động trước đó.
            # - Chưa được lên lịch.
            J_k = []
            for j in range(1, self.N + 2):
                if j not in C and all(i in C for i in self.predecessors[j]):
                    J_k.append(j)
            if not J_k:
                t += 1
                continue

            # Quy tắc chuyển trạng thái
            q = random.random()
            # Kiến chọn hoạt động j thuôc J_k(t) dựa trên hai trường hợp:
            if q <= self.q0:
                values = []
                for j in J_k:
                    values.append((j, (self.tau[t, self.L[j]] ** self.alpha) * (self.eta(t, self.p[j], self.E[j], self.L[j]) ** self.beta)))
                j = max(values, key=lambda x: x[1])[0]
            else:
                pheromone_heuristic = []
                for j in J_k:
                    pheromone_heuristic.append((j, (self.tau[t, j] ** self.alpha) * (self.eta(t, self.p[j], self.E[j], self.L[j]) ** self.beta)))
                total = sum(val for _, val in pheromone_heuristic)
                if total > 0 and np.isfinite(total):
                    probs = []
                    for j, val in pheromone_heuristic:
                        probs.append((j, val / total))
                    j = random.choices([p[0] for p in probs], [p[1] for p in probs])[0]
                else:
                    j = random.choice(J_k)

            # Quy tắc trễ (10% xác suất trì hoãn)
            if self.q1 < q and t <= self.L[j]:
                delay = q * (self.L[j] - t)
                t += int(delay)
                continue

            # Kiểm tra tài nguyên và lên lịch
            if self.check_resource(t, j, solution) and t + self.p[j] <= self.T:
                solution[j] = t
                C.add(j)
                # Cập nhật pheromone cục bộ
                for tt in range(t, t + self.p[j]):
                    if tt < self.T:
                        self.tau[tt, j] = (1 - self.rho) * self.tau[tt, j] + self.rho * self.tau0
            t += 1

        return solution

    def global_update(self, best_solution, best_makespan, prev_best_makespan):
        """Cập nhật pheromone toàn cục"""
        for t in range(self.T):
            for j in range(self.N + 2):
                if best_solution[j] != -1 and t in range(best_solution[j], best_solution[j] + self.p[j]):
                    delta_ms = (1 + max(0, prev_best_makespan - best_makespan)) / best_makespan
                    self.tau[t, j] = (1 - self.delta) * self.tau[t, j] + self.delta * delta_ms
                else:
                    self.tau[t, j] = (1 - self.delta) * self.tau[t, j]

    def dynamic_rule(self, best_solution):
        """Điều chỉnh L_j theo quy tắc động"""
        for j in range(1, self.N + 1):
            S_j = best_solution[j]
            if S_j > self.L[j]:
                self.L[j] = S_j

    def run(self):
        """Chạy thuật toán DDACS"""
        best_solution = None
        best_makespan = float('inf')
        prev_best_makespan = float('inf')

        for _ in range(self.max_iter):
            solutions = []
            for _ in range(self.ant):
                print("=================================================================================")
                solution = self.build_solution()
                if solution[self.N + 1] != -1:
                    makespan = solution[self.N + 1]
                else:
                    makespan = max(solution[j] + self.p[j] for j in range(1, self.N + 1) if solution[j] != -1)
                print(f"Solution: {solution}")
                print(f"Makespan: {makespan}")
                solutions.append((solution, makespan))
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_solution = solution

            # Cập nhật pheromone toàn cục
            self.global_update(best_solution, best_makespan, prev_best_makespan)
            prev_best_makespan = best_makespan

            # Áp dụng quy tắc động
            self.dynamic_rule(best_solution)

        return best_solution, best_makespan

# Ví dụ sử dụng
if __name__ == "__main__":
    # N = 4  # 4 hoạt động + 2 dummy
    # T = 20  # Thời gian tối đa
    # ant = 10  # Số kiến
    # alpha, beta = 1, 2
    # rho, delta = 0.1, 0.1
    # q0 = 0.9
    # max_iter = 50
    # p = [0, 2, 3, 1, 2, 0]  # Thời gian thực hiện (0 và 5 là dummy)
    # R = [4]  # 1 loại tài nguyên, giới hạn 4 đơn vị
    # r = [[0], [2], [3], [1], [2], [0]]  # Yêu cầu tài nguyên
    # predecessors = [[], [0], [0], [1], [2], [3, 4]]  # Quan hệ phụ thuộc
    # successors = [[1, 2], [3], [4], [5], [5], []]

    # Input values for RCPSP based on images (a) and (b)
    N = 10  # Number of real activities (1 to 10), excluding dummy start (0) and end (11)
    T = 20  # Maximum time horizon
    c = 10  # Large enough constant value
    c1 = 50 # Large enough constant value
    ant = 10  # Number of ants
    alpha = 1  # Pheromone importance
    beta = 1  # Heuristic importance
    rho = 0.1  # Pheromone evaporation rate
    delta = 0.1  # Pheromone update parameter
    q0 = 0.9  # Probability of choosing the best option
    q1 = 0.95  # Probability of changing the influence on the decisions of the ants
    max_iter = 50  # Maximum iterations

    # Processing times (p): 0 for dummy activities 0 and 11, others from image (b)
    p = [0, 1, 2, 2, 1, 1, 1, 7, 1, 1, 1, 0]

    # Resource limits (R): From image (b)
    R = [5, 6, 4]  # R1=5, R2=6, R3=4

    # Resource requirements (r): [R1, R2, R3] for each activity, 0s for dummy activities
    r = [
        [0, 0, 0],  # Activity 0 (dummy start)
        [2, 1, 2],  # Activity 1
        [3, 5, 2],  # Activity 2
        [1, 2, 2],  # Activity 3
        [3, 3, 1],  # Activity 4
        [2, 3, 3],  # Activity 5
        [1, 1, 3],  # Activity 6
        [1, 1, 1],  # Activity 7
        [1, 4, 2],  # Activity 8
        [0, 3, 3],  # Activity 9
        [1, 2, 3],  # Activity 10
        [0, 0, 0]  # Activity 11 (dummy end)
    ]

    # Predecessors: Based on image (a)
    predecessors = [
        [],  # 0: No predecessors
        [0],  # 1: Depends on 0
        [0],  # 2: Depends on 0
        [0],  # 3: Depends on 0
        [1],  # 4: Depends on 1
        [1],  # 5: Depends on 1
        [2, 3],  # 6: Depends on 2 and 3
        [4, 5],  # 7: Depends on 4 and 5
        [4, 5],  # 8: Depends on 4 and 5
        [6],  # 9: Depends on 6
        [7, 8, 9],  # 10: Depends on 7, 8, and 9
        [10]  # 11: Depends on 10
    ]

    # Successors: Corresponding to predecessors
    successors = [
        [1, 2, 3],  # 0: Leads to 1, 2, 3
        [4, 5],  # 1: Leads to 4 and 5
        [6],  # 2: Leads to 6
        [6],  # 3: Leads to 6
        [7, 8],  # 4: Leads to 7 and 8
        [7, 8],  # 5: Leads to 7 and 8
        [9],  # 6: Leads to 9
        [10],  # 7: Leads to 10
        [10],  # 8: Leads to 10
        [10],  # 9: Leads to 10
        [11],  # 10: Leads to 11
        []  # 11: No successors
    ]

    ddacs = DDACS(N, T, c, c1, ant, alpha, beta, rho, delta, q0, q1, max_iter, p, R, r, predecessors, successors)
    best_solution, best_makespan = ddacs.run()
    print(f"Best solution: {best_solution}")
    print(f"Makespan: {best_makespan}")
