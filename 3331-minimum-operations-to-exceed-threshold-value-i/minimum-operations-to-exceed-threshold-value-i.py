class Solution(object):
    def minOperations(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """

        answer = 0
        sorted_nums = sorted(nums)
        for n in sorted_nums:
            if n>=k:
                break
            answer += 1
        return answer
        