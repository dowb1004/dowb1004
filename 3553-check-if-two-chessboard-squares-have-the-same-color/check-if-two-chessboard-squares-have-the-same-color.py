class Solution(object):
    def checkTwoChessboards(self, coordinate1, coordinate2):
        """
        :type coordinate1: str
        :type coordinate2: str
        :rtype: bool
        """
        x_1, y_1 = coordinate1
        x_2, y_2 = coordinate2

        if (ord(x_1) + int(y_1)) % 2 == 0 and (ord(x_2) + int(y_2)) % 2 == 0:
            return True
        if (ord(x_1) + int(y_1)) % 2 == 1 and (ord(x_2) + int(y_2)) % 2 == 1:
            return True    

        return False
        