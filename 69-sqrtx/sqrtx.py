class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x == 1:
            return 1
        y = x / 2
        while True:           
            if y * y - x <= 1e-5:
                return int(y)
            y = (y + x / y) / 2.0

        return 0        
        