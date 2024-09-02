class Solution(object):
    def minimumChairs(self, s):
        """
        :type s: str
        :rtype: int
        """
        answer = []
        chair = 0
        for c in s:
            if c == "E":
                chair += 1
            else:
                chair -= 1
            answer.append(chair)

               
        return max(answer)
        