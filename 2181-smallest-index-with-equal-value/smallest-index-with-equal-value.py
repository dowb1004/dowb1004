class Solution(object):
    def smallestEqual(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        answer = -1
        for i, num in enumerate(nums):
            if i % 10 == num:
                answer = i
                break
        return answer         