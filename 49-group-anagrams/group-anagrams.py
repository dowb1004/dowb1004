class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        answer = []
        dic_strs = defaultdict(list)        
        for s in strs:
            dic_strs[''.join(sorted(s, key=lambda x: x, reverse=False))].append(s)
              
        for v in dic_strs.values():
            answer.append(v)
        return answer        
        