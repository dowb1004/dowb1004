class Solution(object):
    def getSneakyNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        c = Counter(nums)
        return [x for x in c if c[x] >= 2]
        