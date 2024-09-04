class Solution(object):
    def sumZero(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        answer = []
        for i in range(n//2):
            answer.append(i+1)
            answer.append(-(i+1))
        if n % 2 == 1:
            answer.append(0)
        return answer
        