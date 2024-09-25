class Solution(object):
    def mostFrequentEven(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        even_num = [x for x in nums if x % 2 == 0]
        if not even_num:
            return -1
        c = Counter(even_num)
        max_freq = max(list(c.values()))
        return min([k for k, v in c.items() if v == max_freq])