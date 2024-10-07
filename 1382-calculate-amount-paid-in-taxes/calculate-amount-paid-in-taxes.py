class Solution(object):
    def calculateTax(self, brackets, income):
        """
        :type brackets: List[List[int]]
        :type income: int
        :rtype: float
        """
        answer = 0
        for i in range(len(brackets)):
            if i == 0:
                prev_upper = 0
            else:
                prev_upper = brackets[i-1][0]
            upper, percent = brackets[i]
            if upper <= income:
                amount = upper - prev_upper                
                answer += (amount * (percent * 0.01))
                
            else:
                amount = income - prev_upper
                answer += (amount * (percent * 0.01))                
                break
        return answer       