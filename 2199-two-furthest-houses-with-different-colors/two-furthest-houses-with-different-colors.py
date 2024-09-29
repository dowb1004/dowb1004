class Solution(object):
    def maxDistance(self, colors):
        """
        :type colors: List[int]
        :rtype: int
        """
        answer = 0
        distance = 0
        for i in range(len(colors)):
            for j in range(i+1, len(colors)):
                if colors[i] != colors[j]:                    
                    distance = max(distance, j-i)
                    answer = distance
        return answer        