class Solution(object):
    def canBeEqual(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        answer = False
        s1_swap1 = s1[2] + s1[1] + s1[0] + s1[3]
        s1_swap2 = s1[0] + s1[3] + s1[2] + s1[1]
        s1_swap3 = s1[2] + s1[3] + s1[0] + s1[1]

        print(s1, s1_swap1, s1_swap2, s1_swap3, s2)
        if s1 == s2 or s1_swap1 == s2 or s1_swap2 == s2 or s1_swap3 == s2:
            answer = True
        
        return answer
        