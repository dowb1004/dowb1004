class Solution(object):
    def oddCells(self, m, n, indices):
        """
        :type m: int
        :type n: int
        :type indices: List[List[int]]
        :rtype: int
        """
        answer = 0
        import numpy as np
        mat = np.zeros((m, n))
        for x, y in indices:
            mat[:, y] += 1
            mat[x] += 1
        for i in range(m):
            for j in range(n):
                if mat[i][j] % 2 == 1:
                    answer += 1
        return answer
        