class Solution(object):
    def kItemsWithMaximumSum(self, numOnes, numZeros, numNegOnes, k):
        """
        :type numOnes: int
        :type numZeros: int
        :type numNegOnes: int
        :type k: int
        :rtype: int
        """
        # 1. k값이 numsOnes 보다 같거나 작을 때
        if k <= numOnes:
            return k
        # 2. k값이 numsOnes 보다 크고 numOnes + numZeros 보다 같거나 작을 때
        elif k > numOnes and k <= numOnes + numZeros:
            return numOnes
        # 3. 그보다 클 때
        else:
            return numOnes + (k-(numOnes + numZeros)) * -1        
        