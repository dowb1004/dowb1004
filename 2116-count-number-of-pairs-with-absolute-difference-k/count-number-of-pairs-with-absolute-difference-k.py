class Solution(object):
    def countKDifference(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        c = Counter(nums)
        answer = 0
        for num in nums:
            if num + k in c:
                answer += c[num + k]
        return answer


        return answer        