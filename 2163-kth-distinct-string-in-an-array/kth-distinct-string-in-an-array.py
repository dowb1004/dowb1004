class Solution(object):
    def kthDistinct(self, arr, k):
        """
        :type arr: List[str]
        :type k: int
        :rtype: str
        """
        c = Counter(arr)
        candidates = [x for x in arr if c[x] == 1]
        print(candidates)
        answer = ""

        if len(candidates) < k:
            return ""
        else:
            answer = candidates[k-1]

        return answer        