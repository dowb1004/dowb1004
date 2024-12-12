class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        for i in range(0, len(haystack)):
            if haystack[i:len(needle)+i] == needle:
                return i
        return -1        