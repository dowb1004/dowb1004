class Solution(object):
    def repeatedCharacter(self, s):
        """
        :type s: str
        :rtype: str
        """
        dic_s = defaultdict(int)
        for c in s:
            dic_s[c] += 1
            if dic_s[c] == 2:
                return c
        return ""