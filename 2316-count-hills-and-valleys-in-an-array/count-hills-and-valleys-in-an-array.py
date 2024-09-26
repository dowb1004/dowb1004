class Solution(object):
    def countHillValley(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 중복은 먼저 제거하고 hill, valley 카운팅 해볼 것
        new_nums = []
        for i in range(len(nums)):
            if i == 0:
                new_nums.append(nums[i])
            elif nums[i] != nums[i-1]:
                new_nums.append(nums[i])
        print(new_nums)

        answer = 0
        for i in range(1, len(new_nums)-1):
            if (new_nums[i-1] < new_nums[i] and new_nums[i+1] < new_nums[i]) or (new_nums[i-1] > new_nums[i] and new_nums[i+1] > new_nums[i]):
                answer += 1
        return answer        
        