class Solution(object):
    def decrypt(self, code, k):
        """
        :type code: List[int]
        :type k: int
        :rtype: List[int]
        """
        answer = []
        sum_code = 0
        for i, c in enumerate(code):
            if k > 0:                                
                for j in range(k):
                    sum_code += code[(i + j + 1) % len(code)]
                answer.append(sum_code)
                sum_code = 0

            elif k < 0:
                for j in range(-k):                    
                    sum_code += code[(i - j - 1) % len(code)]
                answer.append(sum_code)
                sum_code = 0
                    
            else:
                answer.append(0)
        print(answer)
        return answer 
        