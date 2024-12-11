class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        exp = list(set(nums))
        exp = sorted(exp)
        for i in range(len(exp)):
            nums[i] = exp[i]
        
        return len(exp)
        