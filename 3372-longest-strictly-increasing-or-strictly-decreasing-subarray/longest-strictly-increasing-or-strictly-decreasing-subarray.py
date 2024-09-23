class Solution(object):
    def longestMonotonicSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        answer = 1
        inc = 1
        dec = 1

        for i in range(1, len(nums)):
            if nums[i-1] < nums[i]:
                inc += 1                                                    
                dec = 1               
                             
            elif nums[i-1] > nums[i]:
                dec += 1                
                inc = 1
            else:                
                inc = 1
                dec = 1
            answer = max(inc, dec, answer)

        return answer    
        