class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        d = {
            'I': 1,
            'IV': 4,
            'V': 5,
            'IX': 9,
            'X': 10,
            'XL': 40,
            'L': 50,
            'XC': 90,
            'C': 100,
            'CD': 400,
            'D': 500,
            'CM': 900,
            'M': 1000
        }
        i = 0
        answer = 0
        while i < len(s):
            if s[i:i+2] in d:
                answer += d[s[i:i+2]]
                i += 2
            else:
                answer += d[s[i]]
                i += 1 


        # 큰 수부터 매칭해서 더해준다.
        keys = sorted(d.keys(), key=lambda x: d[x], reverse=True)
        print(keys)
        answer = 0
        while s:
            for key in keys:
                if s.startswith(key):
                    # answer에 해당 숫자만큼 더해주고
                    answer += d[key]
                    # s에서 해당 문자열을 잘라내주기
                    s = s[len(key):]
                    break
        return answer
        