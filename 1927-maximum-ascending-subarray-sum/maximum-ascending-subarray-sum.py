class Solution(object):
    def maxAscendingSum(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        answer = 0
        tmp = nums[0] 
        sum_n = 0
        sum_list = []       
        for i in range(1, len(nums)):            
            if tmp < nums[i]:
                sum_n += tmp                
            else:                
                sum_n += tmp                
                sum_list.append(sum_n)
                sum_n = 0
            tmp = nums[i]
        sum_n += tmp
        sum_list.append(sum_n)                    
        answer = max(sum_list)
        return answer
        