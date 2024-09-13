class Solution(object):
    def findTheDistanceValue(self, arr1, arr2, d):
        """
        :type arr1: List[int]
        :type arr2: List[int]
        :type d: int
        :rtype: int
        """
        answer = 0
        flag = 0
        for a1 in arr1:
            for a2 in arr2:
                if abs(a1 - a2) <= d:
                    flag = 1
            if flag == 0:
                answer += 1
            flag = 0    
        return answer
        