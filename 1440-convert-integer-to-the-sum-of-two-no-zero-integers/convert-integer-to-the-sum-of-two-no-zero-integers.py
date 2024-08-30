class Solution(object):
    def getNoZeroIntegers(self, n):
        """
        :type n: int
        :rtype: List[int]
        """

        answer = []
        for i in range(1, n):
            print(list(str(i)), list(str(n-i)))
            if "0" not in list(str(i)) and "0" not in list(str(n-i)):
                answer = [i , n-i]                
                break

        return answer
        