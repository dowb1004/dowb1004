class Solution(object):
    def findColumnWidth(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: List[int]
        """
        answer = []
        len_max = 0
        for j in range(len(grid[0])):
            for i in range(len(grid)):
                len_max = max(len_max, len(str(grid[i][j])))
            answer.append(len_max)
            len_max = 0
        return answer