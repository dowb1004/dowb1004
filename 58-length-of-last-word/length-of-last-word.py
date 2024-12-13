class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        words = s.split(" ")
        for word in words:
            if word != "":
                answer = len(word)
        return answer        
        