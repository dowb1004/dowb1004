class Solution(object):
    def circularGameLosers(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[int]
        """
        # while, set
        cur_loc = 0
        cnt = 0
        visited = set([cur_loc])

        while True:
            cnt += 1
            cur_loc = (cur_loc + (cnt * k)) % n
            if cur_loc in visited:
                break
            visited.add(cur_loc)

        answer = [x+1 for x in range(n) if x not in visited]
        return answer        
        