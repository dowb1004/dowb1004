class Solution(object):
    def numberOfChild(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: int
        """
        return ([x for x in range(n)]+[x for x in range(n)][1:-1][::-1])[k%(2*n-2)]
        