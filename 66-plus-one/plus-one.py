class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        return [int(digit) for digit in str(int("".join(map(str, digits))) + 1)]
        