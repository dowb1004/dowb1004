class Solution(object):
    def isSameAfterReversals(self, num):
        """
        :type num: int
        :rtype: bool
        """
        answer = False
        reverse1_num = str(int(str(num)[::-1]))
        reverse2_num = reverse1_num[::-1]
        
        if str(num) == reverse2_num:
            answer = True
            
        return answer
        